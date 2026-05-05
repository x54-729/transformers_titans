#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the Engram-LaCT hybrid model
Demonstrates different configurations mentioned in README.md
"""

import torch
from configuration_engram_lact import EngramLaCTConfig, EngramConfig
from modeling_lact import LaCTModel

def test_config(name, engram_layer_idx, lact_layer_idx, num_layers=6):
    """Test a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"engram_layer_idx: {engram_layer_idx}")
    print(f"lact_layer_idx: {lact_layer_idx}")

    # Create configuration
    engram_config = EngramConfig(
        tokenizer_name_or_path="deepseek-ai/DeepSeek-V3",
        engram_vocab_size=[129280*5, 129280*5],
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        kernel_size=4,
        hc_mult=4,
        pad_id=2,
        seed=0,
    )

    config = EngramLaCTConfig(
        hidden_size=512,
        num_hidden_layers=num_layers,
        num_attn_heads=8,
        num_lact_heads=4,
        vocab_size=32000,
        engram_config=engram_config,
        engram_layer_idx=engram_layer_idx,
        lact_layer_idx=lact_layer_idx,
    )

    # Create model
    model = LaCTModel(config)

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Count layers by type
    engram_count = len(engram_layer_idx)
    lact_count = len(lact_layer_idx)
    qwen3_count = num_layers - lact_count  # All non-LaCT layers use Qwen3
    engram_qwen3_count = len(set(engram_layer_idx) - set(lact_layer_idx))
    engram_lact_count = len(set(engram_layer_idx) & set(lact_layer_idx))
    pure_qwen3_count = num_layers - len(set(engram_layer_idx) | set(lact_layer_idx))
    pure_lact_count = len(set(lact_layer_idx) - set(engram_layer_idx))

    print(f"\nLayer breakdown:")
    print(f"  - Engram + Qwen3: {engram_qwen3_count} layers")
    print(f"  - Engram + LaCT: {engram_lact_count} layers")
    print(f"  - Pure Qwen3: {pure_qwen3_count} layers")
    print(f"  - Pure LaCT: {pure_lact_count} layers")

    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print(f"\nForward pass successful!")
        print(f"Output shape: {outputs.last_hidden_state.shape}")
    except Exception as e:
        print(f"\nForward pass failed: {e}")

    return model, config


if __name__ == "__main__":
    num_layers = 6

    # Test 1: Pure Qwen3 Attention (both lists empty)
    test_config(
        name="Pure Qwen3 Attention (baseline)",
        engram_layer_idx=[],
        lact_layer_idx=[],
        num_layers=num_layers
    )

    # Test 2: Engram + Attention on all layers
    test_config(
        name="Engram + Qwen3 Attention on all layers",
        engram_layer_idx=list(range(num_layers)),
        lact_layer_idx=[],
        num_layers=num_layers
    )

    # Test 3: Engram + LaCT on all layers
    test_config(
        name="Engram + LaCT on all layers",
        engram_layer_idx=list(range(num_layers)),
        lact_layer_idx=list(range(num_layers)),
        num_layers=num_layers
    )

    # Test 4: Mixed configuration (some Engram+Attention, some pure Attention, some pure LaCT)
    test_config(
        name="Mixed: Engram+Qwen3 (0,1), Pure Qwen3 (2,3), Pure LaCT (4,5)",
        engram_layer_idx=[0, 1],
        lact_layer_idx=[4, 5],
        num_layers=num_layers
    )

    # Test 5: Pure LaCT (no Engram)
    test_config(
        name="Pure LaCT on all layers",
        engram_layer_idx=[],
        lact_layer_idx=list(range(num_layers)),
        num_layers=num_layers
    )

    # Test 6: Alternating Engram+LaCT and Pure Qwen3
    test_config(
        name="Alternating: Engram+LaCT (0,2,4), Pure Qwen3 (1,3,5)",
        engram_layer_idx=[0, 2, 4],
        lact_layer_idx=[0, 2, 4],
        num_layers=num_layers
    )

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")
