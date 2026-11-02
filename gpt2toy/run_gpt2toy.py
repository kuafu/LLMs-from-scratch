"""运行完整 GPT 教学演示，保存终端日志并导出 Markdown / HTML。"""

import os
import sys
from contextlib import redirect_stdout

import torch

from gpt2toy import GPTModel, generate_text_simple
from gpt2toy_data import SimpleTokenizerV2
from gpt2toy_diagnostics import (
    TeachingTrace,
    _TeeWriter,
    analyze_and_print_cross_step_insights,
    collect_and_print_model_info,
)


def main(output_dir=None):
    out_dir = os.fspath(output_dir) if output_dir is not None else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # 自定义微型词表 (与 SimpleTokenizerV2 配合)
    toy_vocab = {
        "<|unk|>": 0,
        "<|endoftext|>": 1,
        "Hello": 2,
        ",": 3,
        "I": 4,
        "am": 5
    }
    tokenizer = SimpleTokenizerV2(toy_vocab)

    TOY_CONFIG = {
        "vocab_size": len(toy_vocab),  # 6
        "context_length": 6,           # 6
        "emb_dim": 4,                  # 4
        "n_heads": 2,                  # 2 (每头维度 head_dim = 2)
        "n_layers": 1,                 # 1
        "drop_rate": 0.0,              # 0.0 (推断模式)
        "qkv_bias": False
    }

    # 初始化固定随机种子的 Toy GPT 模型
    torch.manual_seed(123)
    model = GPTModel(TOY_CONFIG)
    model.eval()  # 禁用 Dropout

    # 1. 打印并收集模型架构、参数构成统计以及全部模块关键权重数据
    param_stats, weights_dict = collect_and_print_model_info(
        model, TOY_CONFIG, tokenizer, print_weights=True
    )

    # 2. 准备输入文本与编码张量
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print("=" * 78)
    print("                         [IN: 输入准备阶段]")
    print("=" * 78)
    print(f"  * 原始输入文本 (start_context) : \"{start_context}\"")
    print(f"  * Tokenizer 编码序列 (IDs)     : {encoded}")
    print(f"  * 逐词映射明细                 : " + ", ".join(f"'{tokenizer.int_to_str[i]}' -> {i}" for i in encoded))
    print(f"  * 封装为 Batch 张量            : {encoded_tensor.tolist()} | 形状: {tuple(encoded_tensor.shape)} (B=1, T={len(encoded)})")
    print("=" * 78)

    # 3. 自回归生成循环 (同时展示模块数据流与每步关键张量/注意力权重数值，并录制快照)
    steps_records = []
    trace = TeachingTrace(tokenizer, steps_records=steps_records)
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=2,
        context_size=TOY_CONFIG["context_length"],
        trace=trace,
    )

    # 4. 跨步因果不变性与同词异位对比分析
    comparisons = analyze_and_print_cross_step_insights(steps_records)

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # 5. 最终生成输出总结
    print("\n" + "=" * 78)
    print("                         [OUT: 最终生成结果]")
    print("=" * 78)
    print(f"  * 最终输出 Token IDs 张量 : {out.tolist()}")
    print(f"  * 最终序列总长度          : {len(out[0])} Tokens")
    print(f"  * 最终反分词还原文本      : \"{decoded_text}\"")
    print("=" * 78 + "\n")

    # 6. 汇总完整剖析数据包并执行双轨导出 (Markdown 深度报告 + 交互式 HTML 观战面板)
    trace_data = {
        "seed": 123,
        "config": {
            **TOY_CONFIG,
            "head_dim": TOY_CONFIG["emb_dim"] // TOY_CONFIG["n_heads"],
            "ffn_hidden_dim": 4 * TOY_CONFIG["emb_dim"],
        },
        "vocab": [
            {"id": idx, "token": word}
            for word, idx in sorted(toy_vocab.items(), key=lambda x: x[1])
        ],
        "start_context": start_context,
        "initial_ids": encoded,
        "final_ids": out[0].tolist(),
        "final_text": decoded_text,
        "param_stats": param_stats,
        "weights": weights_dict,
        "steps": steps_records,
        "comparisons": comparisons,
    }

    from gpt2toy_exporter import export_dual_track_dissection
    md_path, html_path = export_dual_track_dissection(trace_data, out_dir)
    print("=" * 78)
    print("                 [双轨剖析文件已成功生成 (Dual-Track Export)]")
    print("=" * 78)
    print(f"  1. Markdown 深度剖析报告 : {md_path}")
    print(f"  2. HTML 交互式观战面板   : {html_path}")
    print("=" * 78 + "\n")
    return trace_data


def run(output_dir=None):
    """默认输出到脚本目录；验证或独立实验可以指定其他目录。"""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    out_dir = os.fspath(output_dir) if output_dir is not None else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "gpt2toy.txt")
    with open(output_path, "w", encoding="utf-8", buffering=1) as output_file:
        with redirect_stdout(_TeeWriter(sys.stdout, output_file)):
            return main(output_dir=out_dir)


if __name__ == "__main__":
    run()
