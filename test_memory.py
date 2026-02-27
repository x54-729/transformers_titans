import torch
from modeling.neural_memory import BatchNeuralMemory as NeuralMemory
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
config = Qwen3Config(hidden_size=64, num_hidden_layers=6, intermediate_size=192, attn_implementation="eager", num_attention_heads=32,
        num_key_value_heads=8, use_cache=False, pad_token_id=1)

config.titans["update_method"] = "einsum"
config.titans["segment_len"] = 100
config.titans["memory_inter_dim"] = 64*4
config.titans["offload"] = True
config.titans["chunk_size"] = 16
bsz, seqlen, dim, inter_dim = 8, 512, config.hidden_size, config.titans["memory_inter_dim"]
model = NeuralMemory(config).cuda().to(torch.bfloat16)


past_surprise = load("past_surprise", bsz, inter_dim, dim).cuda()
past_memory_param = load("past_memory_param", bsz, inter_dim, dim).cuda()
surprise_update = load("surprise_update", bsz, seqlen, inter_dim, dim).cuda()
eta = load("eta", bsz, seqlen, 1).cuda()
alpha = load("alpha", bsz, seqlen, 1).cuda()

memory_param_n, surprise_n = model.naive_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
memory_param, surprise = model.einsum_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
memory_param_s, surprise_s = model.assocscan_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
print(memory_param_n)
print(memory_param)
print((memory_param_n-memory_param).abs().max())
print((surprise_n-surprise).abs().max())
print((memory_param_s-memory_param).abs().max())
print((surprise_s-surprise).abs().max())
# breakpoint()
# print(surprise_n, surprise)
import time
t = time.time()
for i in range(50):
    memory_param_n, surprise_n = model.naive_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
print("Naive Update time:", time.time() - t)
t = time.time()
for i in range(50):
    memory_param, surprise = model.einsum_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
print("Einsum Update time:", time.time() - t)
t = time.time()
for i in range(50):
    memory_param, surprise = model.assocscan_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
print("AssocScan Update time:", time.time() - t)