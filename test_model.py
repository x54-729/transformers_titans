import torch
from modeling.neural_memory import NeuralMemory
from modeling.modeling_qwen3 import Qwen3ForCausalLM
from modeling.configuration_qwen3 import Qwen3Config
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
config = Qwen3Config(hidden_size=2048, num_hidden_layers=24, intermediate_size=int(2048*2.5), attn_implementation="eager", num_attention_heads=32,
        num_key_value_heads=8, use_cache=False, pad_token_id=1)
bsz, seqlen, dim, inter_dim = 8, 4200, config.hidden_size, config.titans["memory_inter_dim"]
config.titans["update_method"] = "einsum"
config.titans["segment_len"] = 1024
config.titans["chunk_size"] = 64
config.titans["memory_inter_dim"] = config.hidden_size
# config.titans["offload"] = True
x = torch.randint(32, 1239, (bsz, seqlen)).cuda()
model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model.gradient_checkpointing_enable()
print(sum([p.numel() for p in model.parameters()])/1024/1024/1024)
outputs = model(x)
print(outputs.logits.shape)

# for i in range(config.num_hidden_layers):
#     print(f"=========== layer {i}==============")
#     print(model.model.layers[i].neural_memory.state_dict())
# print(model.lm_head.weight)
loss = outputs.loss
loss = outputs.logits.sum()
loss.backward()
optimizer.step()

# for i in range(config.num_hidden_layers):
#     print(f"=========== layer {i}==============")
#     print(model.model.layers[i].neural_memory.state_dict())
# print(model.lm_head.weight)
