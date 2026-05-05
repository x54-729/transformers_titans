# -*- coding: utf-8 -*-

from typing import Optional, Tuple
import torch
import torch.nn as nn

from fla.modules import RMSNorm, RotaryEmbedding
from fla.models.utils import Cache

from .attention_utils import compute_attention


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attn_heads
        self.num_attention_heads = config.num_attn_heads
        self.num_heads = config.num_attn_heads
        self.num_key_value_heads = config.num_attn_heads  # For simplicity, not using GQA
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.qk_norm = config.attn_qk_norm
        self.window_size = config.window_size

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        if self.qk_norm:
            self.q_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(self.head_dim, eps=config.norm_eps)
            self.k_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Rotary embedding
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        batch_size, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_attention_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization if enabled
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Unified attention computation
        attn_output, attn_weights, past_key_values, seqlen_offset, max_seqlen, cu_seqlens = compute_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            layer_idx=self.layer_idx,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            rotary_emb=self.rotary,
            window_size=self.window_size,
            scaling=self.scaling,
            max_position_embeddings=self.config.max_position_embeddings,
            output_attentions=output_attentions,
            use_cache=use_cache,
            dropout=self.attention_dropout,
            training=self.training,
            **kwargs,
        )

        # Project output
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values