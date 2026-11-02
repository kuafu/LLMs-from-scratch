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
    collect_same_token_example,
)


# 随机种子固定同一模型规格的初始化；调整参数形状后，不保证与旧模型权重相同。
SEED = 123
TOY_VOCAB = {
    "<|unk|>": 0,
    "<|endoftext|>": 1,
    "Hello": 2,
    ",": 3,
    "I": 4,
    "am": 5,
}
TOY_CONFIG = {
    "vocab_size": len(TOY_VOCAB),
    "context_length": 5,
    "emb_dim": 4,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
# 生成输入与重复词探针分开，避免把随机生成的词误当作同词异位样本。
START_CONTEXT = "Hello, I am"
REPEATED_CONTEXT = "Hello, I am I"


def main(output_dir=None, *, config=None, vocab=None, start_context=START_CONTEXT,
         repeated_context=REPEATED_CONTEXT, max_new_tokens=2, seed=SEED):
    """组织一次可复现的 GPT 教学实验，导出报告并返回完整 trace_data。

    自回归生成与固定重复词探针使用同一个 eval 模型、分别执行前向传播；
    记录的权重、张量、实际预测及比较结论共同组成 Markdown/HTML 的数据来源。
    """
    out_dir = os.fspath(output_dir) if output_dir is not None else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    toy_vocab = dict(TOY_VOCAB if vocab is None else vocab)
    cfg = dict(TOY_CONFIG if config is None else config)
    if cfg["vocab_size"] != len(toy_vocab):
        raise ValueError("vocab_size 必须与实际词表大小一致")
    tokenizer = SimpleTokenizerV2(toy_vocab)

    # 初始化固定随机种子的 Toy GPT 模型
    torch.manual_seed(seed)
    model = GPTModel(cfg)
    model.eval()  # 禁用 Dropout

    # 1. 打印并收集模型架构、参数构成统计以及全部模块关键权重数据
    param_stats, weights_dict = collect_and_print_model_info(model, cfg, tokenizer, print_weights=True, seed=seed)

    # 2. 准备输入文本与编码张量
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

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
        max_new_tokens=max_new_tokens,
        context_size=cfg["context_length"],
        trace=trace,
    )

    # 4. 固定重复词输入单独前向，不依赖随机生成的词。
    same_token_example = collect_same_token_example(model, tokenizer, repeated_context)
    comparisons = analyze_and_print_cross_step_insights(steps_records, same_token_example)

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
        "seed": seed,
        "config": {
            **cfg,
            "head_dim": cfg["emb_dim"] // cfg["n_heads"],
            "ffn_hidden_dim": model.trf_blocks[0].ff.layers[0].out_features if model.trf_blocks else 0,
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
    """启动完整教学演示，把标准输出同时写入终端和 UTF-8 日志。

    默认输出到脚本目录；验证或独立实验可指定其他目录，返回 main 的剖析数据。
    """
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
