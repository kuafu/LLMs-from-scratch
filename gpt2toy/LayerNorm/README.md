# LayerNorm 技术说明

本文结合 [gpt2toy.py](../gpt2toy.py) 中的 `LayerNorm`，解释均值、方差、归一化和可学习参数。相关逐步打印示例见 [layer_norm_step_by_step.py](../math_breakdown/layer_norm_step_by_step.py)。

## 1. LayerNorm 做什么？

Layer Normalization（层归一化）对一组特征先减去均值，再按其离散程度缩放，最后通过可学习参数调整结果。它帮助控制送入后续网络的特征尺度，通常有利于训练稳定性，但不保证训练一定稳定。

对于一个具有 `D` 个特征的 token，记其输入为 `x = [x₁, ..., xᴰ]`：

\[
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,\qquad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2
\]

\[
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y_i = \gamma_i\hat{x}_i+\beta_i
\]

其中 `μ` 是均值，`σ²` 是总体方差，`ε` 防止除零；`γ` 和 `β` 是逐特征学习的缩放和平移参数。

## 2. 均值和方差的意义

- **均值 `mean`**：这组特征的平均位置。`x - mean` 表示每个特征相对于中心的偏移。
- **方差 `var`**：各偏移量的平方的平均值，描述这组特征的分散程度。平方使正负偏移不会互相抵消，也使较大的偏移有更大的影响。
- **标准差**是 `sqrt(var)`，与输入有相同单位；代码里的归一化除数是 `sqrt(var + eps)`，与严格的标准差略有区别。

这里的总体方差除以 `D`。样本方差通常除以 `D - 1`，是另一种统计量；项目中的 LayerNorm 不使用它。

## 3. 手算一个 token

假设某个 token 的四个特征为：

```text
x = [0.80185723, -0.15227951, 0.21122205, -3.89896703]
```

首先计算均值：

```text
mean = (0.80185723 - 0.15227951 + 0.21122205 - 3.89896703) / 4
     = -0.759541815
```

再计算方差：

| 特征值 | 减去均值 | 偏差的平方 |
| ---: | ---: | ---: |
| 0.80185723 | 1.561399045 | 2.43796698 |
| -0.15227951 | 0.607262305 | 0.36876751 |
| 0.21122205 | 0.970763865 | 0.94238248 |
| -3.89896703 | -3.139425215 | 9.85599068 |

```text
var = (2.43796698 + 0.36876751 + 0.94238248 + 9.85599068) / 4
    ≈ 3.40127691
```

表格中的数值已四舍五入，下列结果按未舍入的中间值计算。取项目的 `eps = 1e-5`：

```text
sqrt(var + eps) ≈ 1.84425782
norm_x ≈ [0.84662731, 0.32927192, 0.52637102, -1.70227025]
```

`norm_x` 的均值约为 0。由于分母加入了 `eps`，其总体方差实际为 `var / (var + eps)`，此例约为 `0.99999706`，接近但不严格等于 1。`scale` 初始化为全 1、`shift` 初始化为全 0 时，初始输出等于 `norm_x`；训练后通常会不同。

## 4. 与项目代码对照

省略 trace 后，[项目实现](../gpt2toy.py)的核心计算是：

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
std = torch.sqrt(var + self.eps)
norm_x = (x - mean) / std
out = self.scale * norm_x + self.shift
```

- `dim=-1`：沿最后一个维度计算。在 `[B, T, D]` 输入中，它是嵌入特征维 `D`。
- `keepdim=True`：保留长度为 1 的末维，便于后续广播。
- `unbiased=False`：方差除以元素个数 `D`；也可写为 `correction=0`。
- `eps = 1e-5`：如果所有特征相同，方差为 0，此项使分母仍为正。
- `scale` 和 `shift` 的形状均为 `[D]`，作为参数参与训练；它们跨 batch 和 token 位置共享，但每个特征各有自己的参数。

归一化后的 `norm_x` 均值约为 0、方差不超过 1；逐特征仿射变换之后，`out` 的均值和方差不再受此约束。若输入所有特征相同，`norm_x` 全为 0，此时输出为 `shift`。

## 5. 张量形状和广播

以 `B=2`、`T=3`、`D=4` 为例：

| 张量 | 形状 | 意义 |
| --- | --- | --- |
| `x` | `[2, 3, 4]` | 两个样本，每个有三个 token，每个 token 有四个特征 |
| `mean`、`var`、`std` | `[2, 3, 1]` | 每个 token 各有一组统计量 |
| `x - mean`、`norm_x` | `[2, 3, 4]` | 对应 token 的统计量广播到四个特征 |
| `scale`、`shift` | `[4]` | 广播到所有样本和位置 |
| `out` | `[2, 3, 4]` | 保持输入形状 |

例如 `mean[0, 1, 0]` 只由 `x[0, 1, :]` 的四个特征计算。不同 token 不共享本次输入的均值或方差，也不会跨 batch 计算；这与共享的 `scale`、`shift` 参数是两回事。

## 6. 在 Transformer 中的位置

当前项目的 `TransformerBlock` 使用 Pre-LN：

```text
x1 = x  + Dropout(Attention(norm1(x)))
x2 = x1 + Dropout(FFN(norm2(x1)))
```

`norm1` 和 `norm2` 是独立的 LayerNorm 实例，参数各自学习。归一化后的值进入子层；残差旁路传递的是归一化之前的输入。所有 Transformer block 结束后，模型还会经过独立的 `final_norm`，再进入输出头。残差流的详细过程见 [残差连接说明](../RESIDUAL_CONNECTIONS.md)。

## 7. 运行验证

在项目目录下运行以下代码，可以检查手算结果，也可以对照 PyTorch 的 `nn.LayerNorm`：

```python
import torch
from gpt2toy import LayerNorm

torch.set_printoptions(precision=8, sci_mode=False)
x = torch.tensor([[[0.80185723, -0.15227951, 0.21122205, -3.89896703]]],
                 dtype=torch.float64)  # [B=1, T=1, D=4]

mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + 1e-5)

toy = LayerNorm(4).double()  # scale=1, shift=0, eps=1e-5
reference = torch.nn.LayerNorm(4, eps=1e-5).double()  # weight=1, bias=0
torch.testing.assert_close(toy(x), reference(x))
torch.testing.assert_close(toy(x), norm_x)

print('mean:', mean)
print('var:', var)
print('norm_x:', norm_x)
```

这里的输入是展示时的小数，使用 `float64` 便于复核手算；原模型通常使用 `float32`，而且打印出来的小数可能已经舍入，因此末位结果可能不同。
