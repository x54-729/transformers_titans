import json
import os
import sys
import time
import argparse
import gc
import shutil

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.nn.functional import all_reduce
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _has_foreach_support,
)

from xtuner.v1.utils.grad_norm import cal_total_norm
from xtuner.v1.utils import get_logger, Config
from xtuner.v1._writer import get_writer
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


from modeling.modeling_qwen3 import Qwen3MACForCausalLM
from modeling.configuration_qwen3 import Qwen3MACConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataloader import InternDataloader, InternDataloaderConfig

def log_format(debug: bool = False, module: str | None = None, rank: int | None = None):
    if rank is None:
        prefix = ""
    else:
        prefix = f"[RANK {rank}]"

    formatter = f"{prefix}[{{time:YYYY-MM-DD HH:mm:ss}}][<level>{{level}}</level>]"

    if module is not None:
        formatter += f"[{module}]"

    if debug:
        formatter += "[<cyan>{name}</cyan>:"
        formatter += "<cyan>{function}</cyan>:"
        formatter += "<cyan>{line}</cyan>]"

    formatter += " <level>{message}</level>"
    return formatter

def init_data_mesh():
    world_size = int(os.getenv("WORLD_SIZE", 1))

    data_mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp",),
    )
    return data_mesh

def init_logger(work_dir):
    log_dir = os.path.join(work_dir, "logs")
    if RANK == 0:
        os.makedirs(log_dir, exist_ok=True)
    dist.barrier()
    # Logging system maybe need better design
    logger = get_logger()
    logger.remove()
    logger.add(os.path.join(log_dir, f"rank{RANK}.log"), format=log_format(), backtrace=True, catch=True)
    logger.add(sys.stderr, format=log_format(rank=RANK))
    return logger, log_dir

# def group_tensors_by_device_mesh_and_placements(tensors):
#     grouped_tensors = {}
#     for tensor in tensors:
#         key = (tensor.device_mesh, tensor.placements)
#         if key in grouped_tensors:
#             grouped_tensors[key].append(tensor)
#         else:
#             grouped_tensors[key] = [tensor]
#     return grouped_tensors

def cal_total_norm(
    tensors, norm_type, foreach=None, dtype=torch.float32
):
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    device = tensors[0].device  # For eg: device(type='cuda', index=0)
    if (foreach is None and _has_foreach_support(tensors, device)) or (  # type: ignore
        foreach and _device_has_foreach_support(device)
    ):
        norms = torch._foreach_norm(tensors, norm_type, dtype=dtype)  # type: ignore
    elif foreach:
        raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
    else:
        norms = tuple(torch.linalg.vector_norm(g, norm_type, dtype=dtype) for g in tensors)

    local_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type, dtype=dtype)
    if norm_type == 2:
        local_norm_squared = local_norm**2
        dist.all_reduce(local_norm_squared, group=data_mesh["dp"].get_group())
        global_norm = local_norm_squared**0.5
    else:
        raise NotImplementedError
    return global_norm

def cal_grad_norm(grads, dtype=torch.float32):
    # grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    # print(f"clip_grad_norm dtype: {dtype}")
    total_norms = []
    # for grads in grouped_grads.values():
    for grad in grads:
        total_norm = cal_total_norm(grads, norm_type=2.0, foreach=True, dtype=dtype)
        total_norms.append(total_norm)
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
    grad_norm = grad_norm.to(grads[0].dtype)
    return grad_norm

def save_dataloader(dataloader, save_path, data_mesh, consumed_samples):
    _gathered_list = [None for _ in range(data_mesh["dp"].size())]
    dist.all_gather_object(_gathered_list, consumed_samples, group=data_mesh["dp"].get_group())
    global_consumed_samples = sum(_gathered_list)  # type: ignore[arg-type]

    # 原来这个 get_state_dict 在 if 里面，之前 xtuner 是怎么正常运行的？
    dataloader_state = dataloader.get_state_dict(global_consumed_samples)
    if RANK == 0:
        torch.save(dataloader_state, save_path)

def load_dataloader(dataloader, save_path):
    dataloader_state = torch.load(save_path, map_location=DEVICE, weights_only=False)
    dataloader.load_state_dict(dataloader_state)

    return dataloader

def find_latest_ckpt(work_dir):
    if not config.RESUME:
        return False
    if isinstance(config.RESUME, str):
        return config.RESUME
    elif not os.path.exists(latest_info_path):
        return False
    else:
        with open(latest_info_path) as fp:
            latest_info = json.load(fp)
        return latest_info["path"]

def resume_model(work_dir, model_class, model_config, tokenizer_path):
    latest_path = find_latest_ckpt(work_dir)
    if not latest_path:
        model_dir = os.path.join(work_dir, "init_model")
        # TODO
        if not os.path.exists(model_dir) and RANK == 0:
            model = model_class(model_config)
            model.save_pretrained(model_dir)
            del model
    else:
        logger.info(f"Resume model from {latest_path}")
        model_dir = os.path.join(latest_path, "model")
        tokenizer_path = model_dir
    dist.barrier()
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=model_config.torch_dtype, trust_remote_code=True, attn_implementation="eager").cuda()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model.train()

    model.gradient_checkpointing_enable()
    logger.info(f"Total Params: {sum([p.numel() for p in model.parameters()])/1024/1024/1024}B")
    model = DDP(model, device_mesh=data_mesh)

    return model, tokenizer
    

def resume_states(work_dir, optimizer, scheduler, dataloader):
    latest_path = find_latest_ckpt(work_dir)
    if not latest_path:
        cur_step = 0
        consumed_samples = 0
        consumed_tokens = 0
    else:
        states_dir = os.path.join(latest_path, "states")

        optimizer_path = os.path.join(states_dir, "optimizer", f"rank{RANK}")
        logger.info(f"Resume optimizer from {optimizer_path}")
        optimizer_states = torch.load(optimizer_path)
        optimizer.load_state_dict(optimizer_states)

        scheduler_path = os.path.join(states_dir, "scheduler")
        logger.info(f"Resume scheduler from {scheduler_path}")
        scheduler_states = torch.load(scheduler_path)
        scheduler.load_state_dict(scheduler_states)

        dataloader_path = os.path.join(states_dir, "dataloader")
        logger.info(f"Resume dataloader from {dataloader_path}")
        dataloader = load_dataloader(dataloader, dataloader_path)

        meta_path = os.path.join(states_dir, "meta", f"rank{RANK}")
        meta = torch.load(meta_path)
        logger.info(f"Resume meta info: {meta}")
        cur_step = meta["step"]
        consumed_samples = meta["consumed_samples"]
        consumed_tokens = meta["consumed_tokens"]

    return optimizer, scheduler, dataloader, cur_step, consumed_samples, consumed_tokens

def save_model_and_states(save_dir, cur_step, consumed_tokens, consumed_samples, model, optimizer, lr_scheduler, dataloader):

    model_dir = os.path.join(save_dir, "model")
    states_dir = os.path.join(save_dir, "states")
    optimizer_dir = os.path.join(states_dir, "optimizer")
    meta_dir = os.path.join(states_dir, "meta")
    scheduler_path = os.path.join(states_dir, "scheduler")
    
    if RANK == 0:
        os.makedirs(states_dir, exist_ok=True)
        os.makedirs(optimizer_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        scheduler_path = os.path.join(states_dir, "scheduler")

        model.module.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # scheduler
        torch.save(lr_scheduler.state_dict(), scheduler_path)
        
    dist.barrier()

    # optimizer
    torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, f"rank{RANK}"))
    # dataloader
    save_dataloader(dataloader, os.path.join(states_dir, "dataloader"), data_mesh, consumed_samples)
    # meta
    meta_info = {
        "step": cur_step,
        "consumed_tokens": consumed_tokens,
        "consumed_samples": consumed_samples
    }
    torch.save(meta_info, os.path.join(meta_dir, f"rank{RANK}"))

    dist.barrier()

    with open(latest_info_path, "w") as fp:
        json.dump({
            "step": cur_step,
            "path": save_dir,
        }, fp, indent=2)
    logger.info(f"Checkpoint Saved to {save_dir}")

parser = argparse.ArgumentParser(description="Train LLM")

parser.add_argument("--config")
args = parser.parse_args()

config = Config.fromfile(args.config)

monkey_patch_hf_modules_cache()

if not dist.is_initialized():
    init_process_group(backend="cpu:gloo,cuda:nccl")
data_mesh = init_data_mesh()
DEVICE = torch.cuda.current_device()
torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 0))
RANK = int(os.getenv("RANK", 0))

if RANK == 0:
    os.makedirs(config.WORK_DIR, exist_ok=True)
dist.barrier()

latest_info_path = os.path.join(config.WORK_DIR, "latest.json")

logger, log_dir = init_logger(config.WORK_DIR)

logger.info(f"================= Config =================")
logger.info(config.pretty_text)
logger.info(f"==========================================")

model, tokenizer = resume_model(config.WORK_DIR, config.model_class, config.model_config, config.TOKENIZER_PATH)

dataloader_cfg = InternDataloaderConfig(
    config_path=config.INTERNLM_CFG,
    seq_len=config.SEQ_LEN,
    internlm_micro_batch_size=1,
    pack_max_length=config.SEQ_LEN,
    global_batch_tokens=config.GLOBAL_BATCH_TOKENS,
    num_worker=2,
)
dataloader = dataloader_cfg.build(
    tokenizer=tokenizer,
    dp_mesh=data_mesh["dp"],
    global_batch_size=config.GLOBAL_BATCH_SIZE,
    micro_batch_size=config.GLOBAL_BATCH_SIZE / data_mesh["dp"].size(),
    seed=config.SEED,
    total_step=config.TOTAL_STEPS,
)

optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
def warmup_fn(x):
    return x / config.WARMUP if x < config.WARMUP else 1

warmup_scheduler = LambdaLR(optimizer, warmup_fn)
scheduler = LambdaLR(optimizer, lambda x: 1.0)
lr_scheduler = SequentialLR(
    optimizer=optimizer,
    schedulers=[warmup_scheduler, scheduler],
    milestones=[config.WARMUP],
)
optimizer, lr_scheduler, dataloader, cur_step, consumed_samples, consumed_tokens = resume_states(config.WORK_DIR, optimizer, lr_scheduler, dataloader)

# check grad norm
torch.autograd.set_detect_anomaly(True)

######################## Training ##################
data_iter = iter(dataloader)

log_dir = os.path.join(config.WORK_DIR, "tensorboard", f"rank{RANK}")
exp_tracker = get_writer(writer_type="tensorboard", log_dir=log_dir)

start_time = time.time()
while cur_step < config.TOTAL_STEPS:
    step_start_time = time.time()
    data_batch = next(data_iter)
    consumed_samples += len(data_batch)
    step_consumed_tokens = torch.tensor(0.0, device=DEVICE)

    input_ids = []
    shifted_labels = []
    attention_mask = []
    cur_max_len = 0
    for data in data_batch:
        seq_ctx = data["seq_ctx"].to(DEVICE)
        # [0, seq_len] or [0, act_seq_len, seq_len] with pad at tail
        assert len(seq_ctx.cu_seq_lens_k) <= 3, seq_ctx.cu_seq_lens_k
        input_ids.append(seq_ctx.input_ids)
        shifted_labels.append(data["shifted_labels"].to(DEVICE))
        attention_mask.append(seq_ctx.mask)

        cur_seq_len = seq_ctx.mask.sum().item()
        step_consumed_tokens += cur_seq_len
        cur_max_len = max(cur_max_len, cur_seq_len)

    del data_batch
    # remove common padding
    input_ids = torch.cat(input_ids, dim=0)[:, :cur_max_len]
    shifted_labels = torch.cat(shifted_labels, dim=0)[:, :cur_max_len]
    attention_mask = torch.cat(attention_mask, dim=0)[:, :cur_max_len]

    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits.float()
    logits = logits.reshape(-1, logits.size(-1))  # (bs * seq_len, vocab_size)
    shifted_labels = shifted_labels.flatten()
    loss = F.cross_entropy(logits, shifted_labels, ignore_index=-100)

    if dist.is_initialized():
        loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD) / WORLD_SIZE

    reduced_loss = loss.detach().item()

    loss.backward()

    if getattr(config, "DEBUG", False):
        per_grad_norm_before = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            per_grad_norm_before[name] = torch.norm(param.grad).item()
            # if torch.isnan(param.grad).any().item():
            #     print(name)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)

    if getattr(config, "DEBUG", False):
        per_grad_norm_after = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            per_grad_norm_after[name] = torch.norm(param.grad).item()

    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        logger.info(f"Gradient norm {grad_norm} is invalid, skipping optimizer step.")
        optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()

    lr_scheduler.step()

    step_end_time = time.time()

    cur_step += 1
    consumed_tokens += step_consumed_tokens

    ############# log step ###############
    lr = lr_scheduler.get_last_lr()[0]

    all_step_tokens_list = [torch.empty(1, device=DEVICE) for i in range(WORLD_SIZE)]
    all_total_tokens = torch.tensor(consumed_tokens, device=DEVICE)
    # logger.info(f"tokens before {step_consumed_tokens}, {consumed_tokens}")
    dist.all_gather(all_step_tokens_list, torch.tensor(step_consumed_tokens, device=DEVICE))
    dist.all_reduce(all_total_tokens)
    # logger.info(f"tokens after {all_step_tokens_list}, {all_total_tokens}")
    all_total_tokens = all_total_tokens.item()
    all_step_tokens = sum(all_step_tokens_list).item()

    step_time = step_end_time - step_start_time
    tgs = step_consumed_tokens / step_time
    total_time = step_end_time - start_time
    logger.info(
        f"[RANK {RANK}] {cur_step}/{config.TOTAL_STEPS} : "
        f"Loss: {reduced_loss:.6f} Grad Norm: {grad_norm:.6e} LR: {lr:.6e} "
        f"Rank Step Tokens: {step_consumed_tokens} Rank Total Tokens: {consumed_tokens} "
        f"Step Tokens: {all_step_tokens} Total Tokens: {all_total_tokens} "
        f"Time: {step_time:.2f}s Total Time: {total_time:.2f}s Tgs: {tgs:.4f} "
        f"Input Shape {input_ids.shape}"
    )
    log_scalars = {
        "loss": reduced_loss,
        "grad_norm": grad_norm.item(),
        "lr": lr,
        "tokens/rank_step": step_consumed_tokens,
        "tokens/rank_total": consumed_tokens,
        "tokens/all_step": all_step_tokens,
        "tokens/all_total": all_total_tokens,
        "speed/step_time": step_time,
        "speed/total_time": total_time,
        "speed/tgs": tgs,
    }

    if getattr(config, "DEBUG", False):
        for name, grad_norm in per_grad_norm_before.items():
            log_scalars[f"grad_norm_before/{name}"] = grad_norm
        for name, grad_norm in per_grad_norm_after.items():
            log_scalars[f"grad_norm_after/{name}"] = grad_norm
    exp_tracker.add_scalars(tag_scalar_dict=log_scalars, global_step=cur_step)

    if cur_step % config.SAVE_FREQ == 0 or cur_step == config.TOTAL_STEPS -1:
        save_dir = os.path.join(os.path.join(config.WORK_DIR, f"step-{cur_step}"))
        save_model_and_states(save_dir, cur_step, consumed_tokens, consumed_samples, model, optimizer, lr_scheduler, dataloader)
    elif cur_step % config.SNAPSHOT == 0:
        save_dir = os.path.join(os.path.join(config.WORK_DIR, f"snapshot"))
        savedir_legacy = os.path.join(os.path.join(config.WORK_DIR, f"snapshot_legacy"))
        if RANK == 0 and os.path.exists(save_dir):
            shutil.move(save_dir, savedir_legacy)
        save_model_and_states(save_dir, cur_step, consumed_tokens, consumed_samples, model, optimizer, lr_scheduler, dataloader)
        if RANK == 0 and os.path.exists(savedir_legacy):
            shutil.rmtree(savedir_legacy)

    # breakpoint()

    if cur_step % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()

logger.info(f"Training Finished. Time Cost: {time.time() - start_time}s")
exp_tracker.close()