"""独立重现 x_emb = tok_embeds + pos_embeds 的全部前置计算。

来源：
  run_gpt2toy.py：随机种子、词表、配置及输入文本。
  gpt2toy_data.py：SimpleTokenizerV2.encode 的分词与 ID 映射逻辑。
  gpt2toy.py：GPTModel 的两张嵌入表初始化及 forward 中的相加计算。

运行（从 gpt2toy 目录）：
  python math_breakdown/embedding_broadcast_add_step_by_step.py

仅依赖 PyTorch 和 Python 标准库，不导入或运行完整 GPT 模型。
"""

import re
import sys

import torch
import torch.nn as nn


def main():
    # 1. 当前教学演示的数据定义；仅保留本等式实际依赖的配置。
    seed = 123
    vocab = {
        "<|unk|>": 0,
        "<|endoftext|>": 1,
        "Hello": 2,
        ",": 3,
        "I": 4,
        "am": 5,
    }
    cfg = {
        "vocab_size": len(vocab),
        "context_length": 5,
        "emb_dim": 4,
    }
    start_context = "Hello, I am"

    # 2. 与 GPTModel.__init__ 相同：设定种子后，先创建词表，再创建位置表。
    # 两张表保存原始 float32 参数，不能用日志中四位小数的值替代。
    # 后续 Transformer 等模块的初始化不影响已经创建好的这两张表。
    torch.manual_seed(seed)
    tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

    # 3. 抽取 SimpleTokenizerV2.encode 的计算：分词、过滤空白、查 token ID。
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', start_context)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    preprocessed = [
        item if item in vocab else "<|unk|>"
        for item in preprocessed
    ]
    encoded = [vocab[item] for item in preprocessed]  # [2, 3, 4, 5]
    input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # (1, 4)

    # 4. 与 GPTModel.forward 相同：得到序列长度，分别查词表与位置表。
    # 本脚本展示前向数值，关闭梯度记录不改变这些计算结果。
    with torch.no_grad():
        _, seq_len = input_ids.shape
        tok_embeds = tok_emb(input_ids)  # (1, 4, 4)
        pos_indices = torch.arange(seq_len, device=input_ids.device)  # (4,)
        pos_embeds = pos_emb(pos_indices)  # (4, 4)

        # 5. 目标等式：右对齐维度，缺失的 batch 维按 1 处理。
        # (1, 4, 4) + (4, 4) -> (1, 4, 4)
        # x_emb[b, t, d] = tok_embeds[b, t, d] + pos_embeds[t, d]
        x_emb = tok_embeds + pos_embeds

    # 6. 打印所有前置数据和计算结果。
    torch.set_printoptions(precision=4, sci_mode=False)
    print("seed:", seed)
    print("vocab:", vocab)
    print("cfg:", cfg)
    print("start_context:", repr(start_context))
    print("tokens:", preprocessed)
    print("encoded:", encoded)
    print("seq_len:", seq_len)
    for name, value in (
        ("input_ids", input_ids),
        ("tok_emb.weight", tok_emb.weight.detach()),
        ("pos_emb.weight", pos_emb.weight.detach()),
        ("tok_embeds", tok_embeds),
        ("pos_indices", pos_indices),
        ("pos_embeds", pos_embeds),
        ("x_emb", x_emb),
    ):
        print(f"\n{name} | shape={tuple(value.shape)}")
        print(value)

    print("\n广播对齐：(1, 4, 4) + (   4, 4)")
    print("逻辑补维：(1, 4, 4) + (1, 4, 4) -> (1, 4, 4)")
    print("\n逐元素计算（显示到八位小数，计算使用原始 float32 值）：")
    for t, token in enumerate(preprocessed):
        print(f"t={t}, token={token!r}, ID={encoded[t]}, Pos={t}")
        for d in range(cfg["emb_dim"]):
            print(
                f"  x_emb[0, {t}, {d}] = "
                f"{tok_embeds[0, t, d].item():.8f} + "
                f"({pos_embeds[t, d].item():.8f}) = "
                f"{x_emb[0, t, d].item():.8f}"
            )

    return x_emb


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
