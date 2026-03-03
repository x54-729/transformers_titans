import os
import time
import gc
import torch
from pydantic import ConfigDict
from typing import Union, Iterator, Dict, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.distributed.tensor.device_mesh import DeviceMesh
import torch.distributed as dist
from collections import defaultdict
import numpy as np

from xtuner.v1.datasets.config import BaseDataloaderConfig
from xtuner.v1.datasets.dataloader import BaseDataloader
from xtuner.v1.datasets.collator import ColateItem
from xtuner.v1.utils import get_logger
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import IGNORE_INDEX
from xtuner.v1.utils.pad import pad_to_max_length
from internlm.utils.execution_time import execution_time_collecter as etc
from internlm.data.train_state import StreamingTrainState
from internlm.data.streaming.weighted_dataset import StreamingWeightedDataset

with etc.collect_execute_time("import_time"):
    from internlm.core.context import global_context as gpc
    from internlm.data.build_dataloader import (
        build_train_loader_with_data_type,
    )
    from internlm.data.train_state import get_train_state
    from internlm.initialize import initialize_distributed_env
    from internlm.train.pipeline import load_new_batch_with_train_state
    from internlm.utils.common import (
        catch_error_node,
    )

from custom_xtuner.data import dump


logger = get_logger()


class InternDataloader(BaseDataloader):
    def __init__(
        self,
        dataloader,
        dp_mesh,
        seed,
        pack_max_length,
        bs_per_iter,
        iters_per_step,
        # pad_token_id,
    ):
        self._dataloader = dataloader
        # dataloader is None for unit test
        self.train_state = (
            get_train_state(dataloader) if dataloader is not None else None
        )
        self._dp_mesh = dp_mesh
        self._dp_size = dp_mesh.size()
        self._dp_rank = dp_mesh.get_local_rank()
        gpc.train_state = self.train_state
        self._seed = seed
        # self.total_step = gpc.config.data.total_steps
        self.pack_max_length = pack_max_length
        self.bs_per_iter = bs_per_iter
        self.iters_per_step = iters_per_step
        self.pad_token_id = 0
        if dataloader is not None:
            #             StreamingWeightedDataset StreamingPackedDataset
            self.pad_token_id = dataloader.dataset.datasets[0].padding_idx
            logger.info(
                f"pad_token_id from internlm dataloader is: {self.pad_token_id}"
            )
        assert IGNORE_INDEX == -100, "IGNORE_INDEX should be -100 in InternDataloader"

    def get_state_dict(self, consumed_samples: int) -> dict:
        assert isinstance(self.train_state, StreamingTrainState)

        # 聚合各个rank上的 self.train_state.data_state_dict
        state_dict_list = [None] * self._dp_size
        dist.all_gather_object(
            state_dict_list,
            self.train_state.data_state_dict,
            group=self._dp_mesh.get_group(),
        )

        # 聚合各个rank上的 dataset_consumed_tokens
        reduced_dataset_consumed_tokens = defaultdict(int)
        for cur_dataset_state_dict in state_dict_list:
            cur_dataset_consumed_tokens = cur_dataset_state_dict[
                "dataset_consumed_tokens"
            ]
            # key is dataset name, value is consumed tokens
            for k, v in cur_dataset_consumed_tokens.items():
                reduced_dataset_consumed_tokens[k] += v

        if self._dp_rank != 0:
            # 将其他rank上的 dataset_consumed_tokens 清空
            self.train_state.data_state_dict["dataset_consumed_tokens"] = defaultdict(
                int
            )
            return {}
        # dp_rank 0 保存聚合后的 dataset_consumed_tokens
        state_dict_list[0]["dataset_consumed_tokens"] = dict(
            reduced_dataset_consumed_tokens
        )
        for i in range(1, len(state_dict_list)):
            state_dict_list[i]["dataset_consumed_tokens"] = {}

        # 日志
        # for dp_i in range(len(state_dict_list)):
        #     logger.info(
        #         f"Get Save state_dict_list[{dp_i}]['multiple_packed_states']: {state_dict_list[dp_i]['multiple_packed_states']}"
        #     )

        # for dp_i in range(len(state_dict_list)):
        #     sd = copy.deepcopy(state_dict_list[dp_i])
        #     sd.pop("multiple_packed_states")
        #     logger.info(f"Get Save state_dict_list[{dp_i}]: {sd}")

        # 更新rank0 的 self.train_state(gpc.train_state)
        self.train_state.data_state_dict["dataset_consumed_tokens"] = (
            reduced_dataset_consumed_tokens
        )

        # logger.info(
        #     f"Save ckpt. Cur batch:{self.train_state.batch_count}, "  # TODO: batch count
        #     f"dataset consumed tokens statistical overview:\n{reduced_dataset_consumed_tokens}"
        # )

        return {
            "train_state": self.train_state.state_dict(),
            "sampler_state": self.train_state.batch_sampler.state_dict(),
            "dataset_state": state_dict_list,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        assert isinstance(self.train_state, StreamingTrainState)
        # 1. train state
        self.train_state.load_state_dict(state_dict["train_state"])
        # 由于老版本xpuyu的 save_ckp 在 batch_count 增加之前，所以需要有下面的加1逻辑
        # self.batch_count = other_stuffs["batch_count"] + 1
        # 但是现在 save_ckp 在 batch_count 增加之后，所以不需要加1逻辑，做如下抵消修改
        self.train_state.batch_count -= 1

        # 2. sampler
        sampler_states = state_dict["sampler_state"]
        self._dataloader.batch_sampler.load_state_dict(sampler_states)
        # track the actual updates of sampler when using weighted sampling
        self.train_state.init_batch_sampler(self._dataloader.batch_sampler)

        # 3.dataset： 包含各个dp_rank的state_dict_list
        state_dict_list = state_dict["dataset_state"]

        if self._dp_rank == 0:
            cur_dataset_consumed_tokens = state_dict_list[0].pop(
                "dataset_consumed_tokens", {}
            )
            self.train_state.data_state_dict["dataset_consumed_tokens"].update(
                cur_dataset_consumed_tokens
            )

        # 日志
        # logger.info(
        #     f"Load state_dict_list[{self._dp_rank}]['multiple_packed_states']: {state_dict_list[self._dp_rank]['multiple_packed_states']}"
        # )
        # sd = copy.deepcopy(state_dict_list[self._dp_rank])
        # sd.pop("multiple_packed_states")
        # logger.info(f"Load state_dict_list[{self._dp_rank}]: {sd}")

        # 3.1 resume时，dp_size没有变化
        if len(state_dict_list) == self._dp_size:
            for k, v in state_dict_list[self._dp_rank].items():
                if k == "dataset_consumed_tokens":
                    continue
                self.train_state.data_state_dict[k] = v
            assert isinstance(self._dataloader.dataset, StreamingWeightedDataset)
            self._dataloader.dataset.load_state_dict(state_dict_list[self._dp_rank])
            return

        # 3.2 resume时，dp_size变化，len(state_dict_list) != self._dp_size
        if state_dict_list[0]["epochs_to_use"]:
            raise NotImplementedError(
                "Cannot resume training if dp_size changed with `epochs_to_use` set."
                " Try set `epochs_to_use` to None."
            )
        multiple_packed_states_group: Dict[str, List[Dict]] = defaultdict(list)
        consumed_samples = defaultdict(int)
        for state_dict in state_dict_list:
            for key, value in state_dict["consumed_samples"].items():
                consumed_samples[key] += value
            for key, value in state_dict["multiple_packed_states"].items():
                multiple_packed_states_group[key].append(value)
        used_epochs = [state_dict["used_epochs"] for state_dict in state_dict_list]
        max_used_epochs = {k: max(d[k] for d in used_epochs) for k in used_epochs[0]}

        for key in list(multiple_packed_states_group.keys()):
            sort_metrics = [
                (
                    state_dict["tokenization_states"]["aggregation_states"][
                        "file_shift"
                    ],
                    state_dict["tokenization_states"]["aggregation_states"][
                        "jsonl_states"
                    ]["line_shift"],
                    state_dict["seq_offset"],
                )
                for state_dict in multiple_packed_states_group[key]
            ]
            multiple_packed_states_group[key] = sorted(
                zip(sort_metrics, multiple_packed_states_group[key]), key=lambda x: x[0]
            )[-1][-1]

        if self._dp_rank == 0:
            state_dict = {
                "rng_state": np.random.RandomState(seed=self._seed).get_state(),
                "multiple_packed_states": multiple_packed_states_group,
                "consumed_samples": consumed_samples,
                "used_epochs": max_used_epochs,
            }
        else:
            state_dict = {
                "rng_state": np.random.RandomState(
                    seed=self._seed + self._dp_rank
                ).get_state(),
                "multiple_packed_states": multiple_packed_states_group,
                "consumed_samples": {},
                "used_epochs": max_used_epochs,
            }
        for k, v in state_dict.items():
            if k == "dataset_consumed_tokens":
                continue
            self.train_state.data_state_dict[k] = v
        self._dataloader.dataset.load_state_dict(state_dict)

    def __iter__(self) -> Iterator[list[ColateItem]]:
        # time_before_internlm = time.time()
        train_iter = iter(self._dataloader)
        while True:
            # for batch in self._dataloader:
            batch, train_iter = load_new_batch_with_train_state(
                train_dl=self._dataloader,
                train_iter=train_iter,
                train_state=self.train_state,
            )
            # time_after_internlm = time.time()
            # internlm_time = time_after_internlm - time_before_internlm

            # 更新 train state 的统计信息
            gpc.config.batch_count = self.train_state.batch_count
            self.train_state.batch_count += 1

            if os.environ.get("DEBUG_DATA_ITERS", "").strip() != "":
                dump(
                    os.environ.get("DEBUG_DATA_DIR", "/tmp/"),
                    self.train_state.batch_count,
                    batch,
                    "internlm",
                )

            ret = self._convert(batch)
            # time_after_convert = time.time()
            # convert_time = time_after_convert - time_after_internlm
            # logger.debug(
            #     f"Get data batch: internlm cost {internlm_time:.2f}s, convert cost {convert_time:.2f}s"
            # )

            yield ret
            # time_before_internlm = time.time()

    def _convert(self, batch):
        ret: list[ColateItem] = []
        inputs, raw_labels = batch

        for _iter in range(self.iters_per_step):
            input_ids_iter = []
            labels_iter = []
            num_tokens_iter = []
            for idx in range(self.bs_per_iter):
                pos = _iter * self.bs_per_iter + idx
                input_ids = inputs["input_ids"][pos : pos + 1]
                labels = raw_labels[pos : pos + 1]
                cu_seq_lens = inputs["cu_seqlens"][pos]
                # internlm dataloader 的 cu_seq_lens 有可能出现前两个元素为0的情况，这里需要处理
                # (Pdb) print(cu_seq_lens)
                # tensor([    0,     0,  1263,  4306,  6125,  6354,  7157,  7665,  8586,  9447,
                #         11046, 12011, 13524, 14893, 15327, 16605, 17629, 18332, 20622, 21726,
                #         23057, 27153, 29012, 29387, 32768], dtype=torch.int32)
                if cu_seq_lens[1].item() == 0:
                    cu_seq_lens = cu_seq_lens[1:]
                num_tokens = cu_seq_lens[1:] - cu_seq_lens[:-1]
                # logger.info(f"inputs:  {input_ids, cu_seq_lens}")
                if (
                    labels[0, -1].item() == IGNORE_INDEX
                    and input_ids[0, -1].item() == self.pad_token_id
                ):
                    pad_len = num_tokens[-1].item()
                    input_ids = input_ids[:, :-pad_len]
                    labels = labels[:, :-pad_len]
                    num_tokens = num_tokens[:-1]
                    if labels.numel() == 0:
                        logger.warning("There appears to be an empry label sequence. Check your dataset.")
                    else:
                        assert not (
                            labels[0, -1].item() == IGNORE_INDEX
                            and input_ids[0, -1].item() == self.pad_token_id
                        ), "Maybe pad cu_seq_lens error"

                if labels.numel() > 0:
                    input_ids_iter.append(input_ids)
                    labels_iter.append(labels)
                    num_tokens_iter.append(num_tokens)

            # if _iter == 0 and self.train_state.batch_count in [6, ]:
            #     print(f"batch_count={self.train_state.batch_count}, _iter = {_iter}, self.iters_per_step = {self.iters_per_step}, self.bs_per_iter = {self.bs_per_iter}")
            #     import torch.distributed as dist; dist.breakpoint()

            # (bs_per_iter, seq_len) -> (1,n)
            input_ids_iter = torch.cat(input_ids_iter, dim=-1)
            labels_iter = torch.cat(labels_iter, dim=-1)
            num_tokens = torch.cat(num_tokens_iter, dim=-1).tolist()

            pack_max_length = self.pack_max_length
            pad_len = pack_max_length - input_ids_iter.shape[-1]

            if pad_len > 0:
                input_ids_iter = pad_to_max_length(
                    input_ids_iter,
                    self.pad_token_id,
                    max_length=pack_max_length,
                    dim=-1,
                )
                labels_iter = pad_to_max_length(
                    labels_iter, IGNORE_INDEX, max_length=pack_max_length, dim=-1
                )
                num_tokens = [0] + num_tokens + [pad_len]
            elif pad_len < 0:
                raise ValueError(
                    f"Internal Error! Packed sample length {input_ids_iter.shape[-1]} is larger than"
                    f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
                )
            else:
                num_tokens = [0] + num_tokens

            cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

            max_len = max(num_tokens)

            seq_ctx = SequenceContext(
                input_ids=input_ids_iter,  # type: ignore
                cu_seq_lens_q=cu_seq_lens,  # type: ignore
                cu_seq_lens_k=cu_seq_lens,  # type: ignore
                max_length_q=max_len,
                max_length_k=max_len,
                num_padding=pad_len,
            )
            # logger.debug(f"Cur batch max_length_q: {max_len}, num_padding: {pad_len}")
            ret.append(
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": labels_iter,
                }
            )
        return ret


def recur_set_data_cfg(data_cfg: dict, key: str, value: any):
    """Old gpc.config.data is a one-level dict.

    When it is converted to a new format, it becomes a nested dict. See internlm's `from tools.print_loaded_config
    import convert_old_conf_to_new_format`
    """
    if key in data_cfg:
        data_cfg[key] = value
    for k, v in data_cfg.items():
        if isinstance(v, dict):
            recur_set_data_cfg(v, key, value)


class InternDataloaderConfig(BaseDataloaderConfig):
    model_config = ConfigDict(
        title="Dataloader config for xtuner",
        extra="allow",
        arbitrary_types_allowed=True,
    )

    config_path: str
    seq_len: int
    internlm_micro_batch_size: int
    pack_max_length: int
    global_batch_tokens: int
    num_worker: int = 0
    pad_token_id: int | None = None  # 为了兼容Trainer的检查逻辑

    def build(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        dp_mesh: DeviceMesh,
        global_batch_size: int,
        micro_batch_size: int,
        seed: int,
        shuffle: bool = True,
        total_step: int | None = None,
    ) -> InternDataloader:
        assert total_step is not None, "total_step should be set"

        self._init_internlm_env_and_config(seed)
        # adjust total_steps
        recur_set_data_cfg(gpc.config.data, "total_steps", total_step)
        recur_set_data_cfg(gpc.config.data, "seq_len", self.seq_len)
        recur_set_data_cfg(gpc.config.data, "valid_packed_length", self.seq_len)
        recur_set_data_cfg(
            gpc.config.data, "max_length_per_sample", self.seq_len
        )  # in tokenizer_wrapper_cfg sub cfg
        recur_set_data_cfg(gpc.config.data, "micro_bsz", self.internlm_micro_batch_size)
        recur_set_data_cfg(gpc.config.data, "num_worker", self.num_worker)
        gpc.config.data.packed_dataset_cfg.packed_length = (
            gpc.config.data.seq_len * gpc.config.data.micro_bsz
        )
        pack_max_length, bs_per_iter, iters_per_step = self._adjust_batch_and_seq(
            dp_mesh.size(), self.global_batch_tokens, self.pack_max_length
        )
        # TODO get data without pack
        self.pack_max_length = pack_max_length
        gpc.config.data.packed_dataset_cfg["no_pack"] = True
        intern_origin_dataloader = self._build_internlm_origin_dataloader(
            dp_mesh.get_local_rank(), dp_mesh.size()
        )
        return InternDataloader(
            intern_origin_dataloader,
            dp_mesh,
            seed,
            self.pack_max_length,
            bs_per_iter,
            iters_per_step,
            # self.pad_token_id,  # pad_token_id由 internlm dataloader 设置
        )

    def _init_internlm_env_and_config(self, seed):
        with etc.collect_execute_time("init_comm_time"):
            catch_error_node(initialize_distributed_env)(
                config=self.config_path, launcher="torch", seed=seed, old_config=True
            )
        assert hasattr(gpc, "config") and gpc.config is not None

    def _adjust_batch_and_seq(
        self, dp_size: int, global_batch_tokens: int = 0, seq_length: int = -1
    ):
        # intra_layer_micro_batch 在 train_engine.train_step 中处理，所以这里设置为 1
        intra_layer_micro_batch = 1

        # ============= The following logic is very important; please verify it carefully. ============= #
        if global_batch_tokens == 0:  # debug
            logger.info(
                "The global_batch_tokens is set to 0, which will default to "
                "using the internal automatic calculation logic of interntrain. "
                "Please ensure that this is correct on your own."
            )

        else:  # 正式训练走此分支
            recur_set_data_cfg(
                gpc.config.data, "global_batch_size", global_batch_tokens
            )
            step_batch_size_without_seqlen = (
                gpc.config.data.micro_bsz * gpc.config.data.seq_len * dp_size
            )
            assert (
                gpc.config.data.global_batch_size % step_batch_size_without_seqlen == 0
            )
            recur_set_data_cfg(
                gpc.config.data,
                "gradient_accumulation",
                gpc.config.data.global_batch_size // step_batch_size_without_seqlen,
            )
            recur_set_data_cfg(
                gpc.config.data, "micro_num", gpc.config.data.gradient_accumulation
            )
            recur_set_data_cfg(
                gpc.config.data, "batch_size", gpc.config.data.gradient_accumulation
            )
            logger.info(
                f"gpc.config.data.global_batch_size = {gpc.config.data.global_batch_size}, gpc.config.data.gradient_accumulation = {gpc.config.data.gradient_accumulation}, gpc.config.data.micro_num = {gpc.config.data.micro_num}, gpc.config.data.batch_size = {gpc.config.data.batch_size}"
            )

        # =========================================================================================== #
        if seq_length == -1:  # debug
            #         32k                        4k              8
            seq_length = gpc.config.data.seq_len * gpc.config.data.micro_bsz
            #          2
            iters_per_step = gpc.config.data.gradient_accumulation
            bs_per_iter = 1

            # (6,4k) intra_layer_micro_batch=2 -> bs_per_iter=2，梯度累加变成 3, 一次运行 (2,4k)
            #  (6,4k) intra_layer_micro_batch=3 -> bs_per_iter=3，梯度累加变成 2, 一次运行 (3,4k)
            # 在这个设置下，一次 forward shape 是 (args.intra_layer_micro_batch, 4k),不会变成 (args.intra_layer_micro_batch, 8k)
            # 如果想变成 8k，则需要指定 args.seq_length=8k，这样才能和之前逻辑对齐

            # TODO: remove intra_layer_micro_batch, bs_per_iter, iters_per_step logic, already handled outside in trainer
            if intra_layer_micro_batch > 1:
                assert iters_per_step % intra_layer_micro_batch == 0, (
                    "The `intra_layer_micro_batch` should be a divisor of `gradient_accumulation`."
                )
                bs_per_iter = intra_layer_micro_batch
                iters_per_step = iters_per_step // intra_layer_micro_batch

            assert iters_per_step >= 1, (
                f"The intra_layer_micro_batch[{intra_layer_micro_batch}] is set incorrectly. Please reset it."
            )
        else:  # 正式训练走此分支
            assert (
                seq_length % (gpc.config.data.seq_len * gpc.config.data.micro_bsz) == 0
            ), (
                f"seq_length = {seq_length}, gpc.config.data.seq_len = {gpc.config.data.seq_len}, gpc.config.data.micro_bsz = {gpc.config.data.micro_bsz}"
            )
            bs_per_iter = seq_length // (
                gpc.config.data.seq_len * gpc.config.data.micro_bsz
            )
            iters_per_step = gpc.config.data.gradient_accumulation // bs_per_iter
            logger.info(
                f"seq_length = {seq_length}, data.seq_len = {gpc.config.data.seq_len}, data.micro_bsz = {gpc.config.data.micro_bsz}, bs_per_iter = {bs_per_iter}, iters_per_step = {iters_per_step}"
            )

            # 假设外部（interlm_cfg）设置为 (grad_acc=4, pack_seq_len=2048), 希望采用 4096(xtuner pack_max_length) 长度训练
            # 那么则调整为 bs_per_iter=2，gradient_accumulation=2
            # 同时在后面处理会自动变成一次 forward 为 (2,4096)
            # 如果又设置了 intra_layer_micro_batch=2，那么 bs_per_iter=4, gradient_accumulation=1（但是这里不会调整，会在TrainEngine.train_step中调整）
            if intra_layer_micro_batch > 1:
                iters_per_step = iters_per_step // intra_layer_micro_batch
                bs_per_iter *= intra_layer_micro_batch

            assert iters_per_step >= 1, (
                f"The seq_length[{seq_length}] is set incorrectly. Please reset it."
            )

        # logger.info(args)
        return seq_length, bs_per_iter, iters_per_step

    def _build_internlm_origin_dataloader(self, dp_rank: int, dp_size: int):
        start_load_data_t = time.time()
        with etc.collect_execute_time("load_data_time"):
            train_dl = build_train_loader_with_data_type(
                data_cfg=gpc.config.data,
                data_rank=dp_rank,
                data_world_size=dp_size,
            )

        gc.collect()

        load_data_cost_time = time.time() - start_load_data_t
        logger.info(f"[Dataset & Dataloader] Cost {load_data_cost_time:.2f}s")
        return train_dl
