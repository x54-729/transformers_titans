import torch
import gc
from modeling.neural_memory import NeuralMemory
from modeling.modeling_qwen3 import Qwen3MACForCausalLM
from modeling.configuration_qwen3 import Qwen3MACConfig
import os

def load(path, *shape):
    if not os.path.exists(path) or flush:
        t = torch.rand(shape)
        torch.save(t, path)
    else:
        t = torch.load(path)

    return t

# TODO memory offload
flush = True
config = Qwen3MACConfig(hidden_size=1536, num_hidden_layers=24, intermediate_size=int(1536*2.5), attn_implementation="eager", num_attention_heads=32,
        num_key_value_heads=8, use_cache=False, pad_token_id=1)
bsz, seqlen, dim, inter_dim = 8, 4200, config.hidden_size, config.titans["memory_inter_dim"]
config.titans["update_method"] = "einsum"
config.titans["segment_len"] = 1024
config.titans["chunk_size"] = 64
config.titans["memory_inter_dim"] = config.hidden_size
# config.titans["offload"] = True

model = Qwen3MACForCausalLM(config).cuda().to(torch.bfloat16)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model.gradient_checkpointing_enable()
print(sum([p.numel() for p in model.parameters()])/1024/1024/1024)
while True:
    print(f"step start {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")
    x = torch.randint(32, 1239, (bsz, seqlen)).cuda()
    print(f"step start2 {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")
    outputs = model(x, labels=x)
    print(outputs.logits.shape)
    print(f"step start3 {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")

    # for i in range(config.num_hidden_layers):
    #     print(f"=========== layer {i}==============")
    #     print(model.model.layers[i].neural_memory.state_dict())
    # print(model.lm_head.weight)
    loss = outputs.loss
    # loss = outputs.logits.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    # for i in range(config.num_hidden_layers):
    #     print(f"=========== layer {i}==============")
    #     print(model.model.layers[i].neural_memory.state_dict())
    # print(model.lm_head.weight)
    torch.cuda.empty_cache()
    gc.collect()
    print(f"step end {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")
    breakpoint()
