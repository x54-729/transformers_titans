"""
测试 Inplace-TTT & Engram 结合实现的关键逻辑
"""

import torch
from configuration_qwen3 import Qwen3Config

def test_config():
    """测试配置是否正确"""
    print("=" * 60)
    print("测试 1: 配置初始化")
    print("=" * 60)

    engram_config = {
        1: {
            'ngram_size_list': [4, 8],
            'n_embed_per_ngram': 512,
            'n_head_per_ngram': 8,
            'tokenizer_name_or_path': 'deepseek-ai/DeepSeek-V3',
            'pad_id': 2,
            'seed': 0,
            'kernel_size': 4,
            'hc_mult': 4,
            'embed_detach': False,
        },
        5: {
            'ngram_size_list': [2, 4, 8],
            'n_embed_per_ngram': 512,
            'n_head_per_ngram': 8,
            'tokenizer_name_or_path': 'deepseek-ai/DeepSeek-V3',
            'pad_id': 2,
            'seed': 0,
            'kernel_size': 4,
            'hc_mult': 4,
            'embed_detach': True,
        }
    }

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        ttt_layers=[0, 1, 2, 5, 6],
        ttt_mode=True,
        ttt_proj=True,
        ttt_target="input_embed",
        engram_config=engram_config,
        engram_embed_proj=False,
    )

    print(f"✓ 配置创建成功")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    print(f"  - ttt_layers: {config.ttt_layers}")
    print(f"  - engram_config keys: {list(config.engram_config.keys())}")
    print(f"  - engram_embed_proj: {config.engram_embed_proj}")

    # 检查 engram_hidden_size 计算
    for layer_id, cfg in engram_config.items():
        ngram_size_list = cfg['ngram_size_list']
        n_embed_per_ngram = cfg['n_embed_per_ngram']
        engram_hidden_size = len(ngram_size_list) * n_embed_per_ngram
        print(f"  - Layer {layer_id} engram_hidden_size: {engram_hidden_size}")

    return config

def test_decoder_layer_init(config):
    """测试 DecoderLayer 初始化逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: DecoderLayer 初始化")
    print("=" * 60)

    from modeling_qwen3 import Qwen3DecoderLayer

    # 测试不同层的初始化
    test_layers = [0, 1, 2, 5, 6, 11]

    for layer_idx in test_layers:
        layer = Qwen3DecoderLayer(config, layer_idx)

        print(f"\nLayer {layer_idx}:")
        print(f"  - has_engram: {layer.has_engram}")
        print(f"  - is_ttt_layer: {layer.is_ttt_layer}")
        print(f"  - has embed_proj: {layer.embed_proj is not None}")
        print(f"  - prev_engram_hidden_size: {layer.prev_engram_hidden_size}")

        # 检查 MLP 的 ttt 参数
        if layer.is_ttt_layer and hasattr(layer.mlp, 'ttt_conv'):
            ttt_conv_channels = layer.mlp.ttt_conv.in_channels
            print(f"  - ttt_conv channels: {ttt_conv_channels}")

            if layer.mlp.ttt_proj is not None:
                ttt_proj_in = layer.mlp.ttt_proj.in_features
                ttt_proj_out = layer.mlp.ttt_proj.out_features
                print(f"  - ttt_proj: {ttt_proj_in} -> {ttt_proj_out}")

    print("\n✓ DecoderLayer 初始化检查完成")

def test_forward_shape():
    """测试 forward 的形状"""
    print("\n" + "=" * 60)
    print("测试 3: Forward 形状检查")
    print("=" * 60)

    # 创建简化的配置用于测试
    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        ttt_layers=[0, 1, 2],
        ttt_mode=True,
        ttt_proj=True,
        ttt_chunk=64,
        ttt_target="input_embed",
        engram_config={
            1: {
                'ngram_size_list': [2, 4],
                'n_embed_per_ngram': 256,
                'n_head_per_ngram': 8,
                'tokenizer_name_or_path': 'deepseek-ai/DeepSeek-V3',
                'pad_id': 2,
                'seed': 0,
                'kernel_size': 4,
                'hc_mult': 4,
                'embed_detach': False,
                'engram_vocab_size': [129280*5, 129280*5],
            }
        },
        engram_embed_proj=False,
    )

    print(f"配置: hidden_size={config.hidden_size}, ttt_chunk={config.ttt_chunk}")
    print(f"Layer 1 engram_hidden_size: {2 * 256} (2 ngrams × 256 embed_per_ngram)")

    from modeling_qwen3 import Qwen3MLP

    # 测试 MLP forward 的形状
    layer_idx = 1
    ttt_hidden_size = 2 * 256  # ngram_size_list 长度 × n_embed_per_ngram
    mlp = Qwen3MLP(config, layer_idx=layer_idx, ttt_hidden_size=ttt_hidden_size)

    batch_size = 2
    seq_len = 128

    # 创建输入
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    t = torch.randn(batch_size, seq_len, ttt_hidden_size)

    print(f"\n输入形状:")
    print(f"  - x: {x.shape} (hidden_states)")
    print(f"  - t: {t.shape} (target_states/engram_embedding)")

    # Forward pass
    try:
        output = mlp(x, t=t)
        print(f"\n输出形状:")
        print(f"  - output: {output.shape}")

        if output.shape == x.shape:
            print(f"\n✓ 输出形状正确 (与输入 x 相同)")
        else:
            print(f"\n✗ 输出形状错误！期望 {x.shape}，得到 {output.shape}")
    except Exception as e:
        print(f"\n✗ Forward 失败: {e}")
        import traceback
        traceback.print_exc()

def test_engram_target_flow():
    """测试 engram_target 的传递逻辑"""
    print("\n" + "=" * 60)
    print("测试 4: engram_target 传递逻辑")
    print("=" * 60)

    print("\n假设场景：12 层模型，第 1 和 5 层有 engram")
    print("期望行为：")
    print("  - 第 0 层：使用 input_embed 作为 ttt_target")
    print("  - 第 1 层：使用自己的 engram embedding")
    print("  - 第 2-4 层：使用第 1 层的 engram embedding")
    print("  - 第 5 层：使用自己的 engram embedding")
    print("  - 第 6-11 层：使用第 5 层的 engram embedding")

    # 模拟 Model forward 的逻辑
    engram_target = None
    engram_layers = {1, 5}

    print("\n模拟执行流程：")
    for layer_idx in range(12):
        has_engram = layer_idx in engram_layers

        # 模拟 _resolve_ttt_target_states
        if engram_target is not None:
            target_for_layer = f"engram_target (from layer {engram_target})"
        else:
            target_for_layer = "input_embed"

        print(f"  Layer {layer_idx}: ", end="")

        if has_engram:
            print(f"has_engram, receives {target_for_layer}, uses own embedding")
            engram_target = layer_idx
        else:
            print(f"no_engram, uses {target_for_layer}")

    print("\n✓ engram_target 传递逻辑验证完成")

if __name__ == "__main__":
    print("开始测试 Inplace-TTT & Engram 实现\n")

    try:
        # 测试 1: 配置
        config = test_config()

        # 测试 2: DecoderLayer 初始化
        test_decoder_layer_init(config)

        # 测试 3: Forward 形状
        test_forward_shape()

        # 测试 4: engram_target 逻辑
        test_engram_target_flow()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
