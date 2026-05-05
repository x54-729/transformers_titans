# -*- coding: utf-8 -*-
"""
Attention utility functions shared between LaCTSWIGLULayer and Qwen3Attention.
Provides common functionality for Flash Attention, KV cache, and sliding window attention.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None


def upad_attention_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    q_len: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
    """
    Unpad input sequences for variable length sequences with padding.

    This function removes padding tokens and prepares inputs for flash_attn_varlen_func.

    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim]
        v: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim]
        attention_mask: Attention mask of shape [batch_size, seq_len]
        q_len: Query sequence length
        num_heads: Number of attention heads (for query)

    Returns:
        Tuple of:
        - Unpaded query
        - Unpaded key
        - Unpaded value
        - Query indices
        - (cu_seqlens_q, cu_seqlens_k): Cumulative sequence lengths
        - (max_seqlen_q, max_seqlen_k): Maximum sequence lengths
    """
    batch_size, seq_len, num_key_value_heads, head_dim = k.shape
    cache_mask = attention_mask[:, -seq_len:]
    seqlens = cache_mask.sum(-1, dtype=torch.int32)
    indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_k = seqlens.max().item()
    cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

    k = index_first_axis(
        k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
    )
    v = index_first_axis(
        v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
    )

    if q_len == seq_len:
        q = index_first_axis(
            q.reshape(batch_size * seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_q = 1
        # There is a memcpy here, that is very bad.
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=q.device
        )
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -q_len:]
        q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

    return (
        q,
        k,
        v,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )


def compute_seqlen_offset(
    past_key_values,
    layer_idx: int,
    q_len: int,
    attention_mask: Optional[torch.Tensor] = None,
    max_position_embeddings: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute sequence length offset for rotary embeddings when using KV cache.

    Args:
        past_key_values: Cache object
        layer_idx: Current layer index
        q_len: Query sequence length
        attention_mask: Optional attention mask
        max_position_embeddings: Maximum position embeddings

    Returns:
        Tuple of (seqlen_offset, max_seqlen)
    """
    seqlen_offset, max_seqlen = 0, q_len

    if past_key_values is not None:
        seqlen_offset = past_key_values.get_seq_length(layer_idx)
        max_seqlen = q_len + seqlen_offset

        if attention_mask is not None:
            # to delimit the offsets of padding tokens
            seqlen_offset = (
                seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
            )
            max_seqlen = q_len + max(seqlen_offset)

    if max_position_embeddings is not None:
        max_seqlen = max(max_seqlen, max_position_embeddings)

    return seqlen_offset, max_seqlen


def update_kv_cache(
    past_key_values,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    q_len: int,
    window_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Update KV cache with new key and value states.

    Args:
        past_key_values: Cache object
        key_states: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        value_states: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
        layer_idx: Current layer index
        q_len: Query sequence length
        window_size: Optional sliding window size

    Returns:
        Tuple of (cached_key, cached_value, cache_has_content)
    """
    if past_key_values is None:
        return key_states, value_states, False

    cache_has_content = past_key_values.get_seq_length(layer_idx) > 0

    # Flatten key and value for caching: [b, s, h, d] -> [b, s, h*d]
    cache_kwargs = {}
    if window_size is not None:
        cache_kwargs["window_size"] = window_size

    k_cached, v_cached = past_key_values.update(
        attn_state=(key_states.flatten(-2, -1), value_states.flatten(-2, -1)),
        layer_idx=layer_idx,
        offset=q_len,
        cache_kwargs=cache_kwargs,
    )["attn_state"]

    if cache_has_content:
        # Use cached k, v
        key_states = k_cached
        value_states = v_cached

    return key_states, value_states, cache_has_content


def apply_flash_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    q_len: int,
    num_heads: int,
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: int,
    window_size: Optional[int] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    Apply Flash Attention with support for padding, variable length sequences, and sliding window.

    Args:
        query_states: Query tensor [batch_size, seq_len, num_heads, head_dim]
        key_states: Key tensor [batch_size, kv_seq_len, num_key_value_heads, head_dim]
        value_states: Value tensor [batch_size, kv_seq_len, num_key_value_heads, head_dim]
        attention_mask: Optional attention mask [batch_size, seq_len]
        q_len: Query sequence length
        num_heads: Number of attention heads
        cu_seqlens: Optional cumulative sequence lengths
        max_seqlen: Maximum sequence length
        window_size: Optional sliding window size
        causal: Whether to apply causal masking

    Returns:
        Attention output tensor [batch_size, seq_len, num_heads * head_dim]
    """
    if flash_attn_func is None:
        raise ImportError(
            "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
        )

    batch_size = query_states.shape[0]

    # Determine window size for sliding window attention
    window_size_tuple = (
        (-1, -1) if window_size is None else (window_size - 1, 0)
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        q, k, v, indices_q, cu_seq_lens, max_seq_lens = upad_attention_input(
            query_states, key_states, value_states, attention_mask, q_len, num_heads
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_q, max_seqlen_k = max_seq_lens

        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size=window_size_tuple,
        )
        attn_output = pad_input(attn_output, indices_q, batch_size, q_len)
    elif cu_seqlens is not None and flash_attn_varlen_func is not None:
        # Variable length attention
        attn_output = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
            window_size=window_size_tuple,
        ).unsqueeze(0)
    else:
        # Standard flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=causal,
            window_size=window_size_tuple,
        )

    return attn_output


def apply_eager_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    window_size: Optional[int] = None,
    dropout: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply eager (non-Flash) attention with optional sliding window.

    Args:
        query_states: Query tensor [batch_size, num_heads, q_len, head_dim]
        key_states: Key tensor [batch_size, num_heads, kv_len, head_dim]
        value_states: Value tensor [batch_size, num_heads, kv_len, head_dim]
        attention_mask: Optional attention mask
        scaling: Attention scaling factor
        window_size: Optional sliding window size
        dropout: Dropout probability
        training: Whether in training mode

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    q_len = query_states.shape[2]
    kv_len = key_states.shape[2]

    # Apply attention mask (causal mask)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :q_len, :kv_len]
        attn_weights = attn_weights + causal_mask

    # Apply sliding window mask if specified
    if window_size is not None and window_size > 0:
        # Create sliding window mask
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=attn_weights.device, dtype=torch.bool),
            diagonal=-window_size + 1
        )
        mask = mask & torch.tril(
            torch.ones(q_len, kv_len, device=attn_weights.device, dtype=torch.bool),
            diagonal=0
        )
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

    # Softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)

    return attn_output, attn_weights


def compute_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[object] = None,
    rotary_emb: Optional[object] = None,
    window_size: Optional[int] = None,
    scaling: Optional[float] = None,
    max_position_embeddings: Optional[int] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    dropout: float = 0.0,
    training: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object], int, int, Optional[torch.Tensor]]:
    """
    Unified attention computation with automatic Flash Attention / eager attention selection.

    This function handles the complete attention pipeline:
    1. Compute sequence offset for rotary embeddings
    2. Apply rotary embeddings
    3. Update KV cache with sliding window support
    4. Choose and apply Flash Attention or eager attention
    5. Return output and updated cache

    Args:
        query_states: Query tensor [batch_size, seq_len, num_heads, head_dim]
        key_states: Key tensor [batch_size, seq_len, num_key_value_heads, head_dim]
        value_states: Value tensor [batch_size, seq_len, num_key_value_heads, head_dim]
        layer_idx: Current layer index
        num_heads: Number of attention heads (query)
        num_key_value_heads: Number of key/value heads
        head_dim: Dimension of each head
        attention_mask: Optional attention mask
        past_key_values: Optional cache object
        rotary_emb: Optional rotary embedding module (must have __call__ method)
        window_size: Optional sliding window size
        scaling: Attention scaling factor (default: 1/sqrt(head_dim))
        max_position_embeddings: Maximum position embeddings
        output_attentions: Whether to output attention weights
        use_cache: Whether to use/update cache
        dropout: Attention dropout probability
        training: Whether in training mode
        **kwargs: Additional arguments (cu_seqlens, etc.)

    Returns:
        Tuple of:
        - attention_output: [batch_size, seq_len, num_heads * head_dim]
        - attention_weights: Optional [batch_size, num_heads, seq_len, kv_len]
        - past_key_values: Updated cache object
        - seqlen_offset: Sequence length offset for rotary embeddings
        - max_seqlen: Maximum sequence length
        - cu_seqlens: Optional cumulative sequence lengths
    """
    batch_size, q_len, _, _ = query_states.shape
    cu_seqlens = kwargs.get("cu_seqlens", None)

    # Default scaling
    if scaling is None:
        scaling = head_dim ** -0.5

    # Step 1: Compute sequence offset
    seqlen_offset, max_seqlen = compute_seqlen_offset(
        past_key_values=past_key_values,# if use_cache else None,
        layer_idx=layer_idx,
        q_len=q_len,
        attention_mask=attention_mask,
        max_position_embeddings=max_position_embeddings,
    )

    # Step 2: Apply rotary embeddings if provided
    if rotary_emb is not None:
        query_states, key_states = rotary_emb(
            query_states,
            key_states,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

    # Step 3: Update KV cache
    # if use_cache and past_key_values is not None:
    if past_key_values is not None:
        key_states, value_states, cache_has_content = update_kv_cache(
            past_key_values=past_key_values,
            key_states=key_states,
            value_states=value_states,
            layer_idx=layer_idx,
            q_len=q_len,
            window_size=window_size,
        )
        # Reshape cached k, v if needed: [b, s, h*d] -> [b, s, h, d]
        if cache_has_content:
            key_states = key_states.view(batch_size, -1, num_key_value_heads, head_dim)
            value_states = value_states.view(batch_size, -1, num_key_value_heads, head_dim)

    # Step 4: Choose and apply attention
    if flash_attn_func is not None and not output_attentions:
        # Use Flash Attention
        attn_output = apply_flash_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            q_len=q_len,
            num_heads=num_heads,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
            causal=True,
        )
        attn_output = attn_output.reshape(batch_size, q_len, num_heads * head_dim)
        attn_weights = None
    else:
        # Use eager attention
        # Transpose for attention computation: [b, s, h, d] -> [b, h, s, d]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat k, v for grouped query attention
        num_key_value_groups = num_heads // num_key_value_heads
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        # Apply eager attention
        attn_output, attn_weights = apply_eager_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            scaling=scaling,
            window_size=window_size,
            dropout=dropout,
            training=training,
        )

        # Reshape output: [b, h, s, d] -> [b, s, h*d]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, num_heads * head_dim)

    return attn_output, attn_weights, past_key_values, seqlen_offset, max_seqlen, cu_seqlens


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for grouped query attention.

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)

    Args:
        hidden_states: Input tensor [batch, num_key_value_heads, seqlen, head_dim]
        n_rep: Number of repetitions

    Returns:
        Repeated tensor [batch, num_key_value_heads * n_rep, seqlen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
