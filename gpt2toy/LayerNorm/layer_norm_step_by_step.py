"""独立复现默认教学演示首个 TransformerBlock 中的 x_norm1 = self.norm1(x)。

运行：python math_breakdown/layer_norm_step_by_step.py
仅依赖 PyTorch；前置计算缩减为两张嵌入表和固定输入 IDs。
LayerNorm 保留 gpt2toy.py 的参数及计算顺序，将 trace 替换为中间值打印。
"""

import torch
import torch.nn as nn


def show(expression, value):
    print(f"\n{expression} | shape={tuple(value.shape)}")
    print(value)


class LayerNorm(nn.Module):
    """每个 token 独立沿最后的特征维归一化，再进行缩放和平移。"""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # mean：每个 token 的 D 个特征的均值，形状 (B, T, 1)，保留末维便于广播。
        mean = x.mean(dim=-1, keepdim=True)
        # var（variance）：各特征与均值之差的平方的平均值，除以 D，形状 (B, T, 1)。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # std：加 eps 后开平方得到的归一化除数，避免除零，形状 (B, T, 1)。
        std = torch.sqrt(var + self.eps)
        # norm_x：每个特征减去所属 token 的均值，再除以 std，形状仍为 (B, T, D)。
        norm_x = (x - mean) / std
        # out：标准化后逐特征乘可学习缩放 scale、加可学习平移 shift，形状 (B, T, D)。
        out = self.scale * norm_x + self.shift

        print(f"\neps = {self.eps}")
        for expression, value in (
            ("self.scale", self.scale),
            ("self.shift", self.shift),
            ("mean = x.mean(dim=-1, keepdim=True)", mean),
            ("var = x.var(dim=-1, keepdim=True, unbiased=False)", var),
            ("std = torch.sqrt(var + self.eps)", std),
            ("x - mean", x - mean),
            ("norm_x = (x - mean) / std", norm_x),
            ("out = self.scale * norm_x + self.shift", out),
        ):
            show(expression, value)
        return out


class Norm1Example(nn.Module):
    """仅保留目标调用所需的 norm1。"""

    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm(4)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        return x_norm1


def main():
    torch.set_printoptions(precision=8, sci_mode=False)

    # 与 run_gpt2toy.py / GPTModel.__init__ 相同的种子、表形状和初始化顺序。
    # 保留完整表形状，才能重现原始 float32 权重；不使用日志中舍入的数值。
    torch.manual_seed(123)
    tok_emb = nn.Embedding(6, 4)
    pos_emb = nn.Embedding(5, 4)
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)  # Hello , I am

    # norm1 的 ones/zeros 初始化不依赖随机数，可省略原模型的 Attention / FFN。
    model = Norm1Example()
    model.eval()
    with torch.no_grad():
        tok_embeds = tok_emb(input_ids)
        pos_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        pos_embeds = pos_emb(pos_indices)
        # 原演示 drop_rate=0.0 且 model.eval()，嵌入 Dropout 的输出等于输入。
        x = tok_embeds + pos_embeds

        show("input_ids", input_ids)
        show("x = tok_embeds + pos_embeds", x)
        x_norm1 = model(x)  # 进入 forward，执行 x_norm1 = self.norm1(x)。
        show("x_norm1 = self.norm1(x)", x_norm1)
    return x, x_norm1


if __name__ == "__main__":
    main()
