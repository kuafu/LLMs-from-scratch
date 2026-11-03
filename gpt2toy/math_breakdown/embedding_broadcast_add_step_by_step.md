# 数学计算过程拆解：词嵌入与位置嵌入的广播相加

本文针对当前项目的实际输入 `[[2, 3, 4, 5]]`，详细解释：

```python
x_emb = tok_embeds + pos_embeds
```

**这一步把形状为 `(1, 4, 4)` 的词嵌入与形状为 `(4, 4)` 的位置嵌入逐元素相加。PyTorch 从右向左对齐维度，将位置嵌入缺失的 batch 维按大小 `1` 处理；位置维、特征维原本就一一对应，最终结果仍是 `(1, 4, 4)`。**

数据依据为 [gpt2toy.txt](../gpt2toy.txt) 中第一次前向传播的输入嵌入阶段，实现依据为 [GPTModel.forward](../gpt2toy.py)。本说明区分日志中已保留四位小数的数据与实际 float32 张量，避免将二次舍入差异误认为计算错误。

## 1. 两个形状分别在描述什么

当前输入文本是 `Hello, I am`，对应四个 token：

| 位置 `t` | Token ID | Token |
| --- | --- | --- |
| 0 | 2 | `Hello` |
| 1 | 3 | `,` |
| 2 | 4 | `I` |
| 3 | 5 | `am` |

当前 `B = 1`、`T = 4`、`D = emb_dim = 4`：

| 张量 | 形状 | 维度含义 |
| --- | --- | --- |
| `input_ids` | `(1, 4)` | 1 条文本，每条有 4 个 token ID |
| `tok_embeds` | `(1, 4, 4)` | 1 条文本 × 4 个 token × 每个向量 4 个分量 |
| `pos_embeds` | `(4, 4)` | 4 个位置 × 每个位置向量 4 个分量 |
| `x_emb` | `(1, 4, 4)` | 1 条文本 × 4 个 token × 每个融合向量 4 个分量 |

`tok_embeds` 的第二个 `4` 是位置数量，第三个 `4` 是特征数量。两者数值恰好相同，含义不同；本例没有交换这两个维度。

源码中的两次查表如下：

```python
tok_embeds = self.tok_emb(input_ids)
pos_indices = torch.arange(seq_len, device=input_ids.device)
pos_embeds = self.pos_emb(pos_indices)
x_emb = tok_embeds + pos_embeds
```

词表权重是 `(6, 4)`，按照形状为 `(1, 4)` 的 ID 张量取行，结果为 `(1, 4, 4)`。位置表权重是 `(5, 4)`，但本次只用位置 `[0, 1, 2, 3]`，查出的结果为 `(4, 4)`，没有使用位置 4 那一行。

位置编号没有 batch 维，因为这组位置向量可以由批次中的所有文本共同使用。

## 2. 用嵌套括号看清实际数据

词嵌入有三层括号。最外面那层表示 batch，里面只有一条文本：

```text
tok_embeds = [                         ← batch 维：1 条文本
    [                                 ← 这条文本的 4 个 token
        [-0.1606, -0.4015,  0.6957, -1.8061],  ← t=0：Hello
        [ 1.8960, -0.1750,  1.3689, -1.6033],  ← t=1：,
        [-0.7849, -1.4096, -0.4076,  0.7953],  ← t=2：I
        [ 0.9985,  0.2212,  1.8319, -0.3378],  ← t=3：am
    ]
]

shape = (1, 4, 4)
```

位置嵌入有两层括号，直接列出 4 个位置的向量：

```text
pos_embeds = [
    [ 0.9624,  0.2492, -0.4845, -2.0929],  ← 位置 0
    [ 1.3111, -0.2199,  0.2190,  0.2045],  ← 位置 1
    [ 0.5146,  0.9938, -0.2587, -1.0826],  ← 位置 2
    [ 0.1036, -2.1996, -0.0885, -0.5612],  ← 位置 3
]

shape = (4, 4)
```

因此，本例的形状差别在于：词嵌入外面有一层 batch 维，位置嵌入没有这一层。

## 3. 广播怎样逐维匹配

### 3.1 从右向左检查

广播相加要求：每对对齐的维度大小相等，或者其中一个大小为 `1`；缺失的前导维度按 `1` 处理。这与 [PyTorch 官方广播说明](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html#general-semantics) 一致。

把两个形状右对齐：

```text
                         batch B    位置 T    特征 D
tok_embeds                  1          4         4
pos_embeds                  —          4         4
                            ↑
                     缺失的前导维度按 1 处理

广播时的逻辑形状：
tok_embeds                  1          4         4
pos_embeds                  1          4         4
-------------------------------------------------
x_emb                       1          4         4
```

| 检查顺序 | 对齐维度 | 比较 | 结果 |
| --- | --- | --- | --- |
| 第 1 步，从最右侧开始 | 特征 `D` | `4` 与 `4` | 相等，分量一一对应 |
| 第 2 步，向左移动 | 位置 `T` | `4` 与 `4` | 相等，位置一一对应 |
| 第 3 步，再向左移动 | batch `B` | `1` 与缺失维度 | 缺失维度按 `1` 处理，可以对应 |

**本例没有把位置向量复制成 4 行，也没有把一个位置向量加给所有 token。位置嵌入本来就有 4 行，每个 token 使用与其位置对应的那一行。**

### 3.2 用 `unsqueeze(0)` 显式表达缺失的维度

下面两种写法的计算结果相同：

```python
# 原始写法：自动对齐维度。
x_emb = tok_embeds + pos_embeds

# 教学写法：显式在最前面增加大小为 1 的维度。
pos_with_batch = pos_embeds.unsqueeze(0)  # (4, 4) -> (1, 4, 4)
x_emb_explicit = tok_embeds + pos_with_batch
```

`unsqueeze(0)` 增加维度，不改变向量中的数值，也不会复制原始数据。原始表达式不要求先调用它；广播也不会将原来的 `pos_embeds.shape` 永久改成 `(1, 4, 4)`，相加之后它仍是 `(4, 4)`。

### 3.3 每一个输出元素究竟读哪里

本例使用的索引规则是：

```text
x_emb[b, t, d] = tok_embeds[b, t, d] + pos_embeds[t, d]
```

其中 `b=0`，`t=0..3`，`d=0..3`。位置嵌入的索引中没有 `b`。

例如：

```text
x_emb[0, 0, 0] = tok_embeds[0, 0, 0] + pos_embeds[0, 0]
x_emb[0, 0, 1] = tok_embeds[0, 0, 1] + pos_embeds[0, 1]
x_emb[0, 2, 3] = tok_embeds[0, 2, 3] + pos_embeds[2, 3]
```

按行理解就是：

```text
tok_embeds[0, 0, :] + pos_embeds[0, :] → x_emb[0, 0, :]  Hello + 位置 0
tok_embeds[0, 1, :] + pos_embeds[1, :] → x_emb[0, 1, :]  ,     + 位置 1
tok_embeds[0, 2, :] + pos_embeds[2, :] → x_emb[0, 2, :]  I     + 位置 2
tok_embeds[0, 3, :] + pos_embeds[3, :] → x_emb[0, 3, :]  am    + 位置 3
```

Token ID 决定词嵌入查表的行号，位置 `t` 决定位置嵌入查表的行号。例如 `Hello` 的 ID 是 `2`，但出现在位置 `0`，所以使用 `tok_emb.weight[2] + pos_emb.weight[0]`。

## 4. 全部 16 个分量的加法

下面把每个 token 的 4 个分量全部展开。表中“手算”严格使用已经显示为四位小数的两个输入；“日志结果”来自原始 float32 张量计算后的四位小数显示。两列不保证完全相同，原因见第 5 节。

### 4.1 `Hello`：位置 `t=0`

| 分量 `d` | 四位小数输入的手算 | 日志结果 |
| --- | --- | --- |
| 0 | `-0.1606 + 0.9624 = 0.8018` | `0.8019` |
| 1 | `-0.4015 + 0.2492 = -0.1523` | `-0.1523` |
| 2 | `0.6957 + (-0.4845) = 0.2112` | `0.2112` |
| 3 | `-1.8061 + (-2.0929) = -3.8990` | `-3.8990` |

日志中的融合向量为 `[0.8019, -0.1523, 0.2112, -3.8990]`。

### 4.2 `,`：位置 `t=1`

| 分量 `d` | 四位小数输入的手算 | 日志结果 |
| --- | --- | --- |
| 0 | `1.8960 + 1.3111 = 3.2071` | `3.2070` |
| 1 | `-0.1750 + (-0.2199) = -0.3949` | `-0.3950` |
| 2 | `1.3689 + 0.2190 = 1.5879` | `1.5879` |
| 3 | `-1.6033 + 0.2045 = -1.3988` | `-1.3988` |

日志中的融合向量为 `[3.2070, -0.3950, 1.5879, -1.3988]`。

### 4.3 `I`：位置 `t=2`

| 分量 `d` | 四位小数输入的手算 | 日志结果 |
| --- | --- | --- |
| 0 | `-0.7849 + 0.5146 = -0.2703` | `-0.2702` |
| 1 | `-1.4096 + 0.9938 = -0.4158` | `-0.4158` |
| 2 | `-0.4076 + (-0.2587) = -0.6663` | `-0.6664` |
| 3 | `0.7953 + (-1.0826) = -0.2873` | `-0.2872` |

日志中的融合向量为 `[-0.2702, -0.4158, -0.6664, -0.2872]`。

### 4.4 `am`：位置 `t=3`

| 分量 `d` | 四位小数输入的手算 | 日志结果 |
| --- | --- | --- |
| 0 | `0.9985 + 0.1036 = 1.1021` | `1.1022` |
| 1 | `0.2212 + (-2.1996) = -1.9784` | `-1.9784` |
| 2 | `1.8319 + (-0.0885) = 1.7434` | `1.7434` |
| 3 | `-0.3378 + (-0.5612) = -0.8990` | `-0.8990` |

日志中的融合向量为 `[1.1022, -1.9784, 1.7434, -0.8990]`。

### 4.5 将四行放回 batch 维

```text
x_emb = [
    [
        [ 0.8019, -0.1523,  0.2112, -3.8990],
        [ 3.2070, -0.3950,  1.5879, -1.3988],
        [-0.2702, -0.4158, -0.6664, -0.2872],
        [ 1.1022, -1.9784,  1.7434, -0.8990],
    ]
]

shape = (1, 4, 4)
```

这一步计算了 `1 × 4 × 4 = 16` 次标量加法，结果保存 16 个数。每个位置仍是 4 维向量；相加没有增加特征数量，也没有在位置之间进行求和。

## 5. 为什么手算 `0.8018`，日志却显示 `0.8019`

源码直接用张量运算，打印才将数值格式化为四位小数。不能将四位小数显示值当成完整的内部数据。

使用当前 `run_gpt2toy.py` 的配置与种子 `123`，在本地 PyTorch `2.14.1+cpu` 中单独重建模型，`Hello` 的第 0 个分量显示到八位小数是：

```text
词嵌入值：   -0.16056378
位置嵌入值：  0.96242100
float32 和：  0.80185723
```

这些八位小数也属于显示值；计算使用的是内部 float32 值。和保留四位小数得到 `0.8019`。如果先把两个输入分别显示成 `-0.1606` 和 `0.9624`，再手算，就得到 `0.8018`。

两条路径分别是：

```text
日志路径：原始值相加 → 结果保留四位小数
手算路径：输入先保留四位小数 → 再相加
```

舍入与加法的顺序不同，末位就可能不同。第 4 节同时保留两种结果，是为了准确解释用户看到的数据。

## 6. 如果 batch 从 1 变成 2，广播会怎样

当前实际 batch 是 `1`，缺失维度补 `1` 后大小已经匹配，无须沿 batch 维进行多份逻辑扩展。下面只改变 batch 数量，帮助理解“广播复用”的含义。

假设有两条长度均为 4 的文本，词嵌入形状变成 `(2, 4, 4)`，位置嵌入仍是 `(4, 4)`：

```text
tok_embeds      (2, 4, 4)
pos_embeds         (4, 4)
逻辑对齐后     (1, 4, 4)
沿 batch 复用  (2, 4, 4)
输出           (2, 4, 4)
```

两条文本使用同一组位置向量：

```text
x_emb[0, t, d] = tok_embeds[0, t, d] + pos_embeds[t, d]
x_emb[1, t, d] = tok_embeds[1, t, d] + pos_embeds[t, d]
```

共享的是位置嵌入；两条文本各自的词嵌入由各自的 token ID 决定。

可以用 `pos_embeds.unsqueeze(0).expand(2, -1, -1)` 显式观察这组逻辑数据。`expand` 可以通过视图共享原始位置数据，沿新增 batch 维的步长为 `0`。广播不要求先用 `repeat` 复制位置数据，但普通 `+` 仍会生成保存相加结果的张量。

## 7. 对齐维度时容易出现的误解

| 写法或理解 | 本例中的含义或后果 |
| --- | --- |
| `tok_embeds + pos_embeds` | 正确，输出 `(1, 4, 4)` |
| `tok_embeds + pos_embeds.unsqueeze(0)` | 正确，显式增加最前面的 batch 维 |
| `tok_embeds + pos_embeds.unsqueeze(-1)` | 右操作数变为 `(4, 4, 1)`，会广播出 `(4, 4, 4)`，不是本例想要的结果 |
| 只用 `pos_embeds[0]` 相加 | 形状上可以广播，但所有 token 都加位置 0 的向量，改变了位置语义 |
| 把 `(4, 4)` 当作 `(B, T)` | 本例位置嵌入的含义是 `(T, D)`，它没有 batch 维 |
| 看到两个维度都为 `4` 就交换它们 | `T` 与 `D` 含义不同，数值相同不会使它们可以任意交换 |

PyTorch 根据形状对齐，不会根据变量名自动识别“哪一维是 batch”。因此，`unsqueeze` 的位置必须与实际语义一致。

## 8. 可运行验证：使用原始精度重现当前结果

在项目 `gpt2toy` 目录启动 Python，运行以下完整代码。代码直接创建模型并计算嵌入，不调用教学运行器的 `main()`，不重新导出或覆盖现有报告。

```python
import torch
from gpt2toy import GPTModel
from run_gpt2toy import SEED, TOY_CONFIG

torch.manual_seed(SEED)
model = GPTModel(TOY_CONFIG).eval()
torch.set_printoptions(precision=4, sci_mode=False)

with torch.no_grad():
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    tok_embeds = model.tok_emb(input_ids)
    pos_embeds = model.pos_emb(torch.arange(input_ids.shape[1]))

    # 两种写法必须得到完全相同的结果。
    x_emb = tok_embeds + pos_embeds
    explicit = tok_embeds + pos_embeds.unsqueeze(0)
    torch.testing.assert_close(x_emb, explicit, rtol=0, atol=0)

    # 用逐元素索引表达同一个计算，核对全部 16 个元素。
    by_index = torch.empty_like(x_emb)
    for b in range(tok_embeds.shape[0]):
        for t in range(tok_embeds.shape[1]):
            for d in range(tok_embeds.shape[2]):
                by_index[b, t, d] = tok_embeds[b, t, d] + pos_embeds[t, d]
    torch.testing.assert_close(x_emb, by_index, rtol=0, atol=0)

    print("tok_embeds:", tuple(tok_embeds.shape))
    print("pos_embeds:", tuple(pos_embeds.shape))
    print("x_emb:", tuple(x_emb.shape))
    print(x_emb)

    # 两条不同文本，验证 batch 维共享同一组位置向量。
    ids_two = torch.tensor([[2, 3, 4, 5], [5, 4, 3, 2]])
    tok_two = model.tok_emb(ids_two)
    x_two = tok_two + pos_embeds
    for b in range(2):
        torch.testing.assert_close(
            x_two[b], tok_two[b] + pos_embeds, rtol=0, atol=0
        )

    shared_pos = pos_embeds.unsqueeze(0).expand(2, -1, -1)
    assert shared_pos.stride(0) == 0
    assert shared_pos.data_ptr() == pos_embeds.data_ptr()
    torch.testing.assert_close(x_two, tok_two + shared_pos, rtol=0, atol=0)
    assert tuple(pos_embeds.shape) == (4, 4)
    assert tuple(x_two.shape) == (2, 4, 4)
    print("广播、显式补维、逐元素索引与 batch 复用验证通过")
```

前半部分应输出：

```text
tok_embeds: (1, 4, 4)
pos_embeds: (4, 4)
x_emb: (1, 4, 4)
tensor([[[ 0.8019, -0.1523,  0.2112, -3.8990],
         [ 3.2070, -0.3950,  1.5879, -1.3988],
         [-0.2702, -0.4158, -0.6664, -0.2872],
         [ 1.1022, -1.9784,  1.7434, -0.8990]]])
```

这些数值对应当前模型配置、初始化顺序和种子；调整模型或运行环境后应重新核对。广播规则与这个特定数值快照相互独立。

## 9. 在模型中的作用

词嵌入按 token ID 提供表示，位置嵌入按位置提供表示。两者在同一个 4 维特征空间中相加，后续网络接收到融合后的向量。每个分量的具体作用由训练学习，不是人工规定某个维度只存词义、另一个维度只存位置。

这一行完成后，源码再执行 `x = self.drop_emb(x_emb)`。本文所有结果都对应相加完成时的 `x_emb`；当前演示配置的 `drop_rate=0.0`，运行器也将模型设为 `eval()`。

可继续阅读：[Token Embedding 查表计算拆解](token_embedding_step_by_step.md)。
