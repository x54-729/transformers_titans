from typing import Callable
import math
import os
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from torch import Tensor
from transformers.activations import ACT2FN
from transformers.utils.import_utils import is_torch_greater_or_equal
if is_torch_greater_or_equal("2.8.0"):
    from torch._higher_order_ops.associative_scan import associative_scan
else:
    associative_scan = None
# associative_scan = None

def debug_print(*args, **kwargs):
    if int(os.getenv("DEBUG", 0)):
        print(*args, **kwargs)

# @torch.jit.script
def binary_operator(
    a: tuple[Tensor, Tensor],
    b: tuple[Tensor, Tensor]
):
    """
    二元操作符，用于 associative_scan 函数。

    该操作符对输入的两个元组进行操作：
    1. 对第一个张量执行逐元素相乘。
    2. 对第二个张量执行逐元素累加乘积（addcmul）。

    参数:
        a (Tuple[torch.Tensor, torch.Tensor]): 第一个输入元组，包含两个张量。
        b (Tuple[torch.Tensor, torch.Tensor]): 第二个输入元组，包含两个张量。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 操作后的结果元组，包含两个张量。
    """
    # 解包第一个输入元组
    a_i, kv_i = a
    # 解包第二个输入元组
    a_j, kv_j = b
    # 返回操作后的结果元组
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)

class Memory(nn.Module):
    """
    MLP Memory Module

    memory_inter_dim: intermediate size of MLP
    memory_layers: num layers of Memory. Usually set to 2.
    memory_act: activateion function.
    memory_bias: whethe set bias for Linear module.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.inter_dim = config.titans["memory_inter_dim"]
        self.num_layers = config.titans["memory_layers"]
        self.bias = config.titans["memory_bias"]
        self.act_fn = ACT2FN[config.titans["memory_act"]]

        assert self.num_layers >= 2
        
        module_list = []
        module_list.append(nn.Linear(self.dim, self.inter_dim, bias=self.bias))
        for i in range(self.num_layers - 2):
            module_list.append(nn.Linear(self.inter_dim, self.inter_dim, bias=self.bias))
        module_list.append(nn.Linear(self.inter_dim, self.dim, bias=self.bias))
        self.weights = nn.ModuleList(module_list)

        self.layer_norm = nn.RMSNorm(self.dim, config.rms_norm_eps)

        self.init_memory()

    def init_memory(self):
        """
        Initialize memory weight
        """
        for name, param in self.named_parameters():
            if "layer_norm" not in name:
                nn.init.xavier_uniform_(param)

    def get_memory_params(self, batch_size):
        memory_params = {}
        for name, param in self.named_parameters():
            expand_shape = [batch_size] + [-1] * len(param.shape)
            # B, *shape
            memory_params[name] = param.unsqueeze(0).expand(*expand_shape)

        return memory_params

    def forward(
        self,
        x
    ):
        """
        Should not be called directly.
        TODO
        """
        x_residual = x
        for i, layer in enumerate(self.weights):
            x = layer(x)
            if i < self.num_layers - 1:
                # TTTMLP gelu
                x = self.act_fn(x)

        # Layer Norm
        x = self.layer_norm(x)
        # x + LN(f_{mlp}(x)) in TTT
        x = x_residual + x

        return x


class NeuralMemory(nn.Module):
    """
    Update and retrieve memory from Memory Module with chunk size and adaptive params.

    chunk_size: mini-batch of parallel TTT
    base_lr: base learning rate of adaptive lr according to TTT
    bias: whether set bias for qkv.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.bias = config.titans["bias"]
        self.chunk_size = config.titans["chunk_size"] # mini-batch of TTT parallel update
        self.base_lr = config.titans["base_lr"]
        self.qkv_act_fn = ACT2FN[config.titans["qkv_act"]]
        self.update_method = config.titans["update_method"]

        # Q, K, V
        self.wq = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )
        self.wk = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )
        self.wv = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )

        # QK-Norm
        self.q_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # adaptive alpha, theta, eta in memory update
        # According to TTT: sigmoid(linear(x)) * lr
        self.alpha = nn.Linear(self.hidden_size, 1) # gate
        self.theta = nn.Linear(self.hidden_size, 1) # lr
        self.eta = nn.Linear(self.hidden_size, 1) # momentum

        # Memory Module
        self.memory = Memory(config)

        # Memory loss and grad function
        def forward_and_loss(params, inputs, target, loss_weights):
            # forward
            # B, D
            pred = functional_call(self.memory, params, inputs)
            # 计算损失，默认为均方误差损失
            loss =  (((target - pred).pow(2)) * loss_weights).mean(dim=-1).sum() # simple mse loss in paper - eq (12) - |M(k) - v|² TODO
            return loss

        # 对每个样本计算梯度
        self.per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims=(None, 1, 1, 0))

    def get_memory_params(self):

        return self.memory.get_memory_params()

    def forward(self, x, memory_params, past_surprises=None):
        """
        Update memory and retreve memory from the newest weights
        """

        bsz, seq_len, _ = x.shape
        num_chunks = math.ceil(seq_len / self.chunk_size)
        for i in range(num_chunks):
            # update memory with mini-batch (chunk_size)
            x_chunk = x[:, i*self.chunk_size:(i+1)*self.chunk_size]
            memory_params, past_surprises = self.store(x_chunk, memory_params, past_surprises)
        output = self.retrieve(x, memory_params)

        return output, memory_params, past_surprises

    def naive_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory using for-loop (naive method)
        """
        seq_len = surprise_update.shape[0]
        # if len(past_memory_param.shape) == 1:
        eta = eta.squeeze(-1)
        alpha = alpha.squeeze(-1)
        for idx in range(seq_len):
            # S_t = eta_t * S_{t-1} - theta_t * u_t
            past_surprise = past_surprise * eta[idx:idx+1] - surprise_update[idx]
            # M_t = (1 - alpha_t) * M_{t-1} + S_t
            past_memory_param = (1 - alpha[idx:idx+1]) * past_memory_param + past_surprise
        return past_memory_param, past_surprise

    def einsum_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory with batch matmul
        S_t = eta_t * ... eta_1 * past_surprise - \sum_{i=1}^{t-1} eta_t * ... eta_{i+1} * theta_{i} * u_{i} - theta_{t} * u_{t}
        M_t = beta_t * ... beta_1 * past_memory + \sum_{i = 1}^{t-1} beta_t * ... beta_{i+1} * S_i + S_t which beta_i = 1 - alpha_i
        """
        if len(past_surprise.shape) == 2:
            weight_shape = "ad"
        else:
            weight_shape = "a"
        seq_len = surprise_update.shape[0]
        eta_matrix = eta.expand(seq_len, seq_len)
        triu_mask_diag = torch.triu(torch.ones_like(eta_matrix, dtype=torch.bool), diagonal=0)
        triu_mask = torch.triu(torch.ones_like(eta_matrix, dtype=torch.bool), diagonal=1)
        # 1,
        # eta_2, 1
        # ...
        # eta_t, eta_t, ..., 1
        eta_matrix = eta_matrix.masked_fill(triu_mask_diag, 1)
        eta_matrix = torch.cumprod(eta_matrix, dim=-2)
        # 1, 0, ...
        # eta_2, 1
        # ...
        # eta_2*...*eta_t, eta_3*...*eta_t, ..., 1
        eta_matrix = eta_matrix.masked_fill(triu_mask, 0)
        # eta_1*S0, ..., eta_1*...*eta_t*S0
        past_surprise_matrix = torch.einsum(f"mn,{weight_shape}->m{weight_shape}", torch.cumprod(eta, dim=-2), past_surprise)
        # \sum_{i=1}^{t-1} eta_t * ... eta_{i+1} * theta_{i} * u_{i} + theta_{t} * u_{t}
        surprise_matrix = torch.einsum(f"mn,n{weight_shape}->m{weight_shape}", eta_matrix, surprise_update)
        surprise_matrix = past_surprise_matrix - surprise_matrix

        beta = 1 - alpha
        # beta_1*...*beta_t*M_{t-1}
        if len(past_memory_param.shape) == 3:
            past_memory_matrix = past_memory_param * torch.cumprod(beta, dim=-2)[-1:]
        else:
            # norm
            past_memory_matrix = past_memory_param * torch.cumprod(beta, dim=-2)[-1]
        # beta_2*...*beta_t, beta_3*...*beta_t, ..., beta_t
        beta_cumprod = torch.cumprod(beta[1:, 0].flip(dims=[0]), dim=-1)
        beta_cumprod = beta_cumprod.flip(dims=[0])
        # \sum_{i = 1}^{t-1} beta_t * ... beta_{i+1} * S_i + S_t
        memory_matrix = torch.einsum(f"m,m{weight_shape}->{weight_shape}", beta_cumprod, surprise_matrix[:-1])
        memory_param = past_memory_matrix + memory_matrix + surprise_matrix[-1]

        return memory_param, surprise_matrix[-1]

    def assocscan_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory with associative scan
        """
        raise NotImplementedError
        # S0, -u1, -u2, -u3, ..., -ut
        surprise_update = torch.cat([past_surprise.unsqueeze(1), -surprise_update], dim=1)
        # 1, eta1, eta2, eta3, ..., eta4
        eta = torch.cat([torch.tensor([[1]], device=eta.device, dtype=eta.dtype), eta], dim=0)
        _, surprise_update = associative_scan(binary_operator, (eta, surprise_update), dim=0)

    def store(self, x, memory_params, past_surprises=None):
        """
        Upate Memory Weights with theta_k, theta_v for current chunk

        x: current input chunk
        memory_params: Memory weights
        surprise: Last chunk's final surprise
        """
        # seq_len <= chunk_size
        bsz, seq_len, _ = x.shape
        if past_surprises is None:
            past_surprises = {}
        
        # Get gate, lr, momentum
        # sum at batch_size dim TODO L, 1
        alpha = self.alpha(x).sum(dim=0).sigmoid()
        theta = self.theta(x).sum(dim=0).sigmoid() * self.base_lr
        eta = self.eta(x).sum(dim=0).sigmoid()

        # k_t with act and l2 norm
        k_proj = self.wk(x)
        k_proj = self.qkv_act_fn(k_proj)
        k_proj = self.k_norm(k_proj)
        # v_t
        v_proj = self.wv(x)
        v_proj = self.qkv_act_fn(v_proj)

        # Per sample Per token memory grad
        # grads: dict, value shape: seq_len, inter_dim, hidden_size
        grads = self.per_sample_grad_fn(memory_params, k_proj, v_proj, theta)

        new_memory_params, new_suprises = {}, {}

        for param_name, surprise_update in grads.items():
            if param_name not in past_surprises:
                # First token's surprise init as 0 TODO
                past_surprises[param_name] = torch.zeros(*surprise_update.shape[1:], device=surprise_update.device, dtype=surprise_update.dtype)
            past_surprise = past_surprises[param_name]
            past_memory_param = memory_params[param_name]

            if self.update_method == "naive":
                # update memory using for-loop (naive method)
                memory_param, surprise = self.naive_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            elif self.update_method == "einsum":
                # update memory using einsum
                memory_param, surprise = self.einsum_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            elif self.update_method == "assocscan":
                # update memory using associative scane
                memory_param, surprise = self.assocscan_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            else:
                raise NotImplementedError(f"Not Implemented: {self.update_method}")
            new_memory_params[param_name] = memory_param
            new_suprises[param_name] = surprise

        return new_memory_params, new_suprises

    def retrieve(self, x, memory_params):
        """
        Get memory from Memory Module with theta_q
        """
        # q_t with act and l2 norm
        q_proj = self.wq(x)
        q_proj = self.qkv_act_fn(q_proj)
        q_proj = self.q_norm(q_proj)
        # get memory
        memory = functional_call(self.memory, memory_params, q_proj)

        return memory
    

class BatchNeuralMemory(nn.Module):
    """
    Update and retrieve memory from Memory Module with chunk size and adaptive params.
    The memory and srprises are updated with a whole chunk (mini-batch) instead of token by token.

    chunk_size: mini-batch of parallel TTT
    base_lr: base learning rate of adaptive lr according to TTT
    bias: whether set bias for qkv.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.bias = config.titans["bias"]
        self.chunk_size = config.titans["chunk_size"] # mini-batch of TTT parallel update
        self.base_lr = config.titans["base_lr"]
        self.qkv_act_fn = ACT2FN[config.titans["qkv_act"]]
        self.update_method = config.titans["update_method"]

        # Q, K, V
        self.wq = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )
        self.wk = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )
        self.wv = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.bias
        )

        # QK-Norm
        self.q_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # adaptive alpha, theta, eta in memory update
        # According to TTT: sigmoid(linear(x)) * lr
        self.alpha = nn.Linear(self.hidden_size * self.chunk_size, 1, bias=False) # gate
        self.theta = nn.Linear(self.hidden_size * self.chunk_size, 1, bias=False) # lr
        self.eta = nn.Linear(self.hidden_size * self.chunk_size, 1, bias=False) # momentum

        # Memory Module
        self.memory = Memory(config)

        # Memory loss and grad function
        def forward_and_loss(params, inputs, target, loss_weights):
            # forward
            # C, D
            pred = functional_call(self.memory, params, inputs)
            # 计算损失，默认为均方误差损失
            loss =  (((target - pred).pow(2)) * loss_weights).mean(dim=-1).sum() # simple mse loss in paper - eq (12) - |M(k) - v|² TODO
            return loss

        # 对每个样本计算梯度
        self.per_sample_grad_fn = vmap(vmap(grad(forward_and_loss), in_dims=(None, 0, 0, 0)), in_dims=(0, 0, 0, 0))

        def retrieve_fn(params, inputs):
            pred = functional_call(self.memory, params, inputs)
            return pred
        
        self.retrieve_fn = vmap(retrieve_fn, in_dims=(0, 0))
        self.assocscan_fn = associative_scan

    def get_memory_params(self, batch_size):

        return self.memory.get_memory_params(batch_size)

    def forward(self, x, memory_params, past_surprises=None):
        """
        Update memory and retreve memory from the newest weights
        """

        memory_params, past_surprises = self.store(x, memory_params, past_surprises)
        output = self.retrieve(x, memory_params)

        return output, memory_params, past_surprises

    def naive_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory using for-loop (naive method)
        """
        bsz, num_chunks = surprise_update.shape[:2]
        if len(past_memory_param.shape) == 3:
            eta = eta.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
        for idx in range(num_chunks):
            # S_t = eta_t * S_{t-1} - theta_t * u_t
            past_surprise = past_surprise * eta[:, idx] - surprise_update[:, idx]
            # M_t = (1 - alpha_t) * M_{t-1} + S_t
            past_memory_param = (1 - alpha[:, idx]) * past_memory_param + past_surprise
        return past_memory_param, past_surprise

    def einsum_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory with batch matmul
        S_t = eta_t * ... eta_1 * past_surprise - \sum_{i=1}^{t-1} eta_t * ... eta_{i+1} * theta_{i} * u_{i} - theta_{t} * u_{t}
        M_t = beta_t * ... beta_1 * past_memory + \sum_{i = 1}^{t-1} beta_t * ... beta_{i+1} * S_i + S_t which beta_i = 1 - alpha_i
        """
        bsz, num_chunks = surprise_update.shape[:2]
        eta_matrix = eta.expand(bsz, num_chunks, num_chunks)
        triu_mask_diag = torch.triu(torch.ones_like(eta_matrix, dtype=torch.bool), diagonal=0)
        triu_mask = torch.triu(torch.ones_like(eta_matrix, dtype=torch.bool), diagonal=1)
        # 1,
        # eta_2, 1
        # ...
        # eta_t, eta_t, ..., 1
        eta_matrix = eta_matrix.masked_fill(triu_mask_diag, 1)
        eta_matrix = torch.cumprod(eta_matrix, dim=-2)
        # 1, 0, ...
        # eta_2, 1
        # ...
        # eta_2*...*eta_t, eta_3*...*eta_t, ..., 1
        eta_matrix = eta_matrix.masked_fill(triu_mask, 0)
        # eta_1*S0, ..., eta_1*...*eta_t*S0
        past_surprise_matrix = torch.einsum(f"bmn,b...->bm...", torch.cumprod(eta, dim=-2), past_surprise)
        # \sum_{i=1}^{t-1} eta_t * ... eta_{i+1} * theta_{i} * u_{i} + theta_{t} * u_{t}
        surprise_matrix = torch.einsum(f"bmn,bn...->bm...", eta_matrix, surprise_update)
        surprise_matrix = past_surprise_matrix - surprise_matrix

        beta = 1 - alpha
        # beta_1*...*beta_t*M_{0}
        if len(past_memory_param.shape) == 3:
            past_memory_matrix = past_memory_param * torch.cumprod(beta, dim=-2)[:, -1:]
        else:
            # norm
            past_memory_matrix = past_memory_param * torch.cumprod(beta, dim=-2)[:, -1]
        # beta_2*...*beta_t, beta_3*...*beta_t, ..., beta_t
        beta_cumprod = torch.cumprod(beta[:, 1:, 0].flip(dims=[1]), dim=-1)
        beta_cumprod = beta_cumprod.flip(dims=[1])
        # \sum_{i = 1}^{t-1} beta_t * ... beta_{i+1} * S_i + S_t
        memory_matrix = torch.einsum(f"bm,bm...->b...", beta_cumprod, surprise_matrix[:, :-1])
        memory_param = past_memory_matrix + memory_matrix + surprise_matrix[:, -1]

        return memory_param, surprise_matrix[:, -1]

    def assocscan_update(self, past_surprise, past_memory_param, surprise_update, eta, alpha):
        """
        Update memory with associative scan
        """
        bsz, num_chunks = surprise_update.shape[:2]
        if len(past_memory_param.shape) == 3:
            eta = eta.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            ones_shape = (1, 1)
        else:
            ones_shape = (1, )
        # S0, -u1, -u2, -u3, ..., -ut
        surprise_update = torch.cat([past_surprise.unsqueeze(1), -surprise_update], dim=1)
        # 1, eta1, eta2, eta3, ..., eta4
        eta = torch.cat([torch.ones(bsz, 1, *ones_shape, device=eta.device, dtype=eta.dtype), eta], dim=1)
        # S0, S1, S2, ...., St
        _, surprise = self.assocscan_fn(binary_operator, (eta, surprise_update), dim=1, combine_mode="generic")

        # M0, S1, S2, S3
        memory_param = torch.cat([past_memory_param.unsqueeze(1), surprise[:, 1:]], dim=1)
        # 1, 1-alpha1, ..., 1-alphat
        beta = torch.cat([torch.ones(bsz, 1, *ones_shape, device=alpha.device, dtype=alpha.dtype), 1-alpha], dim=1)
        _, memory_param = self.assocscan_fn(binary_operator, (beta, memory_param), dim=1, combine_mode="generic")

        return memory_param[:, -1], surprise[:, -1]

    def store(self, x, memory_params, past_surprises=None):
        """
        Upate Memory Weights with theta_k, theta_v chunk by chunk

        x: input
        memory_params: Memory weights
        surprise: Last chunk's final surprise
        """
        bsz, seq_len, hidden_size = x.shape
        if past_surprises is None:
            past_surprises = {}

        # reshape and pad input
        num_chunks = math.ceil(seq_len / self.chunk_size)
        pad_seq_len = self.chunk_size * num_chunks
        if pad_seq_len > seq_len:
            # pad to full
            # TODO pad token id
            x = F.pad(x, (0, 0, 0, pad_seq_len-seq_len), value=self.config.pad_token_id)

        # B, N, C, D
        x = x.view(bsz, num_chunks, self.chunk_size, hidden_size)
        
        # Get gate, lr, momentum for every batch and chunk
        # B, N, 1
        # B, N, C*D
        x_c = x.view(bsz, num_chunks, self.chunk_size*hidden_size)
        alpha = self.alpha(x_c).sigmoid()
        theta = self.theta(x_c).sigmoid() * self.base_lr
        eta = self.eta(x_c).sigmoid()

        # k_t with act and l2 norm
        k_proj = self.wk(x)
        k_proj = self.qkv_act_fn(k_proj)
        k_proj = self.k_norm(k_proj)
        # v_t
        v_proj = self.wv(x)
        v_proj = self.qkv_act_fn(v_proj)

        # Per sample Per token memory grad
        # grads: dict, value shape: bsz, num_chunks, inter_dim, hidden_size
        debug_print(f"before grad {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")
        grads = self.per_sample_grad_fn(memory_params, k_proj, v_proj, theta)
        debug_print(f"after grad {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")

        new_memory_params, new_suprises = {}, {}

        for param_name, surprise_update in grads.items():
            if param_name not in past_surprises:
                # First token's surprise init as 0 TODO
                past_surprises[param_name] = torch.zeros(bsz, *surprise_update.shape[2:], device=surprise_update.device, dtype=surprise_update.dtype)
            past_surprise = past_surprises[param_name]
            past_memory_param = memory_params[param_name]
            debug_print(f"before update {param_name}: {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")

            if self.update_method == "naive":
                # update memory using for-loop (naive method)
                memory_param, surprise = self.naive_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            elif self.update_method == "einsum":
                # update memory using einsum
                memory_param, surprise = self.einsum_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            elif self.update_method == "assocscan":
                # update memory using associative scane
                memory_param, surprise = self.assocscan_update(past_surprise, past_memory_param, surprise_update, eta, alpha)
            else:
                raise NotImplementedError(f"Not Implemented: {self.update_method}")
            new_memory_params[param_name] = memory_param
            new_suprises[param_name] = surprise
            debug_print(f"after update {param_name}: {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")

        debug_print(f"store end: {torch.cuda.memory.memory_allocated()/1024/1024/1024}B")

        return new_memory_params, new_suprises

    def retrieve(self, x, memory_params):
        """
        Get memory from Memory Module with theta_q
        """
        # q_t with act and l2 norm
        q_proj = self.wq(x)
        q_proj = self.qkv_act_fn(q_proj)
        q_proj = self.q_norm(q_proj)
        # get memory
        memory = self.retrieve_fn(memory_params, q_proj)

        return memory
