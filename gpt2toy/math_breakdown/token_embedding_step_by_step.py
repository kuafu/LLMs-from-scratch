import sys
import io
import torch
import torch.nn as nn

# 保证 Windows 终端输出 UTF-8 中文字符正常显示
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

"""
================================================================================
【数学计算过程拆解】Token Embedding: self.tok_emb(input_ids)
================================================================================
输入 (Input IDs):
    input_ids = tensor([[2, 3, 4, 5]])
    形状: [batch_size=1, seq_len=4]

权重矩阵 (Embedding Weights):
    W 形状: [vocab_size=6, emb_dim=4]

数学定义:
    对于输入中的每个 token 索引 idx:
    输出向量 y_i = W[idx]
    
    在严格线性代数意义上，查表等价于：
    y_i = one_hot(idx, vocab_size) @ W
================================================================================
"""

def main():
    torch.manual_seed(123)
    tok_emb = nn.Embedding(6, 4)
    print("tok_emb:", tok_emb)
    print(tok_emb.weight)
    # tok_emb.weight = torch.tensor([
    #     [ 0.3374, -0.1778, -0.3035, -0.5880],  # 索引 0
    #     [ 0.3486,  0.6603, -0.2196, -0.3792],  # 索引 1
    #     [-0.1606, -0.4015,  0.6957, -1.8061],  # 索引 2
    #     [ 1.8960, -0.1750,  1.3689, -1.6033],  # 索引 3
    #     [-0.7849, -1.4096, -0.4076,  0.7953],  # 索引 4
    #     [ 0.9985,  0.2212,  1.8319, -0.3378]   # 索引 5
    # ])

    input_ids = torch.tensor([2, 3, 4, 5], dtype=torch.long).unsqueeze(0)
    print("input_ids")
    print(input_ids)
    #tensor([[2, 3, 4, 5]])

    print("---\n")
    tok_embeds = tok_emb(input_ids)
    print("tok_embeds")
    print(tok_embeds)
    # tensor([[[-0.1606, -0.4015,  0.6957, -1.8061],
    #      [ 1.8960, -0.1750,  1.3689, -1.6033],
    #      [-0.7849, -1.4096, -0.4076,  0.7953],
    #      [ 0.9985,  0.2212,  1.8319, -0.3378]]])


if __name__ == "__main__":
    main()
