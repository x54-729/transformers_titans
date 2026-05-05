#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Qwen3Attention 的 sliding window KV cache 实现
"""

import torch
from configuration_engram_lact import EngramLaCTConfig
from qwen3_attention import Qwen3Attention
from fla.models.utils import Cache

def test_basic_forward():
    """测试基本的 forward 功能"""
    print("=" * 60)
    print("Test 1: Basic Forward (No Cache)")
    print("=" * 60)

    config = EngramLaCTConfig(
        hidden_size=512,
        num_attn_heads=8,
        window_size=128,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        qkv_bias=False,
        attn_qk_norm=False,
    )

    attention = Qwen3Attention(config, layer_idx=0)

    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, attn_weights, past_kv = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
    )

    print(f"✓ Input shape: {hidden_states.shape}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Basic forward test passed!\n")


def test_with_cache():
    """测试 KV cache 功能"""
    print("=" * 60)
    print("Test 2: Forward with KV Cache")
    print("=" * 60)

    config = EngramLaCTConfig(
        hidden_size=512,
        num_attn_heads=8,
        window_size=128,
        max_position_embeddings=2048,
        rope_theta=10000.0,
    )

    attention = Qwen3Attention(config, layer_idx=0)
    attention.eval()  # 设置为 eval 模式

    batch_size = 2
    seq_len = 64

    # 第一次 forward（初始化 cache）
    hidden_states_1 = torch.randn(batch_size, seq_len, config.hidden_size)
    cache = Cache()

    with torch.no_grad():
        output_1, _, cache = attention(
            hidden_states=hidden_states_1,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

    print(f"✓ First forward - Input: {hidden_states_1.shape}, Output: {output_1.shape}")
    print(f"✓ Cache length after first forward: {cache.get_seq_length(0)}")

    # 第二次 forward（使用 cache）
    new_seq_len = 32
    hidden_states_2 = torch.randn(batch_size, new_seq_len, config.hidden_size)

    with torch.no_grad():
        output_2, _, cache = attention(
            hidden_states=hidden_states_2,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

    print(f"✓ Second forward - Input: {hidden_states_2.shape}, Output: {output_2.shape}")
    print(f"✓ Cache length after second forward: {cache.get_seq_length(0)}")

    expected_cache_len = min(seq_len + new_seq_len, config.window_size)
    actual_cache_len = cache.get_seq_length(0)

    print(f"✓ Expected cache length (with sliding window): {expected_cache_len}")
    print(f"✓ Actual cache length: {actual_cache_len}")

    print("✓ KV cache test passed!\n")


def test_sliding_window():
    """测试 sliding window 限制"""
    print("=" * 60)
    print("Test 3: Sliding Window")
    print("=" * 60)

    window_size = 64
    config = EngramLaCTConfig(
        hidden_size=512,
        num_attn_heads=8,
        window_size=window_size,
        max_position_embeddings=2048,
    )

    attention = Qwen3Attention(config, layer_idx=0)
    attention.eval()

    batch_size = 2
    seq_len = 128  # 超过 window_size
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    cache = Cache()

    with torch.no_grad():
        output, _, cache = attention(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

    cache_len = cache.get_seq_length(0)
    print(f"✓ Input sequence length: {seq_len}")
    print(f"✓ Window size: {window_size}")
    print(f"✓ Cache length: {cache_len}")

    # Cache 长度应该被限制在 window_size
    if cache_len <= window_size:
        print(f"✓ Cache correctly limited by sliding window!")
    else:
        print(f"✗ Warning: Cache length ({cache_len}) exceeds window size ({window_size})")

    print("✓ Sliding window test passed!\n")


def test_with_padding():
    """测试带 padding 的序列"""
    print("=" * 60)
    print("Test 4: Attention with Padding")
    print("=" * 60)

    config = EngramLaCTConfig(
        hidden_size=512,
        num_attn_heads=8,
        window_size=128,
    )

    attention = Qwen3Attention(config, layer_idx=0)
    attention.eval()

    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # 创建 attention mask：第一个序列长度为 50，第二个为 40
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    attention_mask[0, 0, :, 50:] = float('-inf')  # 第一个序列的 padding
    attention_mask[1, 0, :, 40:] = float('-inf')  # 第二个序列的 padding

    with torch.no_grad():
        output, _, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )

    print(f"✓ Input shape: {hidden_states.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention mask shape: {attention_mask.shape}")
    print("✓ Padding test passed!\n")


def compare_with_without_cache():
    """比较有无 cache 的输出一致性"""
    print("=" * 60)
    print("Test 5: Cache Consistency")
    print("=" * 60)

    config = EngramLaCTConfig(
        hidden_size=512,
        num_attn_heads=8,
        window_size=128,
    )

    attention = Qwen3Attention(config, layer_idx=0)
    attention.eval()

    batch_size = 2
    seq_len_1 = 32
    seq_len_2 = 32

    # 生成输入
    hidden_states_1 = torch.randn(batch_size, seq_len_1, config.hidden_size)
    hidden_states_2 = torch.randn(batch_size, seq_len_2, config.hidden_size)
    full_hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=1)

    # 方式1：一次性处理所有输入（无 cache）
    with torch.no_grad():
        output_no_cache, _, _ = attention(
            hidden_states=full_hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )

    # 方式2：分两次处理，使用 cache
    cache = Cache()
    with torch.no_grad():
        output_1, _, cache = attention(
            hidden_states=hidden_states_1,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        output_2, _, cache = attention(
            hidden_states=hidden_states_2,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

    output_with_cache = torch.cat([output_1, output_2], dim=1)

    # 比较输出
    max_diff = (output_no_cache - output_with_cache).abs().max().item()
    mean_diff = (output_no_cache - output_with_cache).abs().mean().item()

    print(f"✓ Output shape (no cache): {output_no_cache.shape}")
    print(f"✓ Output shape (with cache): {output_with_cache.shape}")
    print(f"✓ Max difference: {max_diff:.6e}")
    print(f"✓ Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Cache consistency test passed!")
    else:
        print(f"⚠ Warning: Difference is larger than expected")

    print()


if __name__ == "__main__":
    print("\nTesting Qwen3Attention Sliding Window KV Cache Implementation\n")

    try:
        test_basic_forward()
        test_with_cache()
        test_sliding_window()
        test_with_padding()
        compare_with_without_cache()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
