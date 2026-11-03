# torch.arange：按固定步长创建一维张量

`torch.arange()` 从起点开始，按固定步长生成一维张量，**包含起点，不包含终点**。

```python
import torch

x = torch.arange(5)
print(x)        # tensor([0, 1, 2, 3, 4])
print(x.shape)  # torch.Size([5])
print(x.ndim)   # 1
```

`torch.Size([5])` 表示这个一维张量有 5 个元素；这里的 `5` 是元素数量，不是维度数量。

## 1. 三种常见调用形式

```python
torch.arange(5)         # tensor([0, 1, 2, 3, 4])
torch.arange(2, 6)      # tensor([2, 3, 4, 5])
torch.arange(1, 10, 2)  # tensor([1, 3, 5, 7, 9])
```

| 调用形式 | 含义 |
| --- | --- |
| `torch.arange(end)` | 从 `0` 开始，步长为 `1` |
| `torch.arange(start, end)` | 指定起点和终点，步长为 `1` |
| `torch.arange(start, end, step)` | 同时指定起点、终点和步长 |

- `start`：起点，默认是 `0`。
- `end`：终点，必须提供，生成的序列不包含这个值。
- `step`：步长，默认是 `1`，不能为 `0`。

例如 `torch.arange(1, 10, 2)` 依次生成 `1、3、5、7、9`，再加 `2` 就超过终点，因此停止。

## 2. 负数步长和小数步长

步长为负数时，序列递减：

```python
torch.arange(5, 0, -1)
# tensor([5, 4, 3, 2, 1])
```

这里包含 `5`，不包含 `0`。步长方向需要与起点、终点匹配，例如从 `5` 递减到 `0` 应使用负步长。

步长也可以是小数：

```python
torch.arange(0, 1, 0.25)
# tensor([0.0000, 0.2500, 0.5000, 0.7500])
```

浮点数有精度误差，像 `0.1` 这样的步长可能影响端点附近的结果。生成索引时优先使用整数；需要指定采样点数量时，可考虑 `torch.linspace(start, end, steps=数量)`，它会包含两端点（`steps > 1` 时）。

## 3. dtype：元素的数据类型

没有显式指定 `dtype`，且 `start`、`end`、`step` 都是整数时，结果默认是 `torch.int64`：

```python
x = torch.arange(4)
print(x.dtype)  # torch.int64
```

可以显式指定为浮点类型：

```python
x_float = torch.arange(4, dtype=torch.float32)
print(x_float)        # tensor([0., 1., 2., 3.])
print(x_float.dtype)  # torch.float32
```

如果任一范围参数是浮点数，且未指定 `dtype`，结果使用当前默认浮点类型，通常是 `torch.float32`；该默认值可以通过 `torch.set_default_dtype()` 修改。

## 4. device：创建张量的位置

`device` 可以指定张量创建在哪个设备上。处理已有输入时，常用输入张量的设备：

```python
input_ids = torch.tensor([[10, 20, 30, 40]])
seq_len = input_ids.shape[1]
positions = torch.arange(seq_len, device=input_ids.device)

print(positions)         # tensor([0, 1, 2, 3])
print(positions.shape)   # torch.Size([4])
print(positions.device)  # cpu
```

本例的 `input_ids` 在 CPU 上，因此位置编号也创建在 CPU 上。模型参数与输入也应放在相同设备上。

## 5. 在 GPT 中生成位置编号

项目的 [GPTModel.forward](../gpt2toy/gpt2toy.py#L243) 使用：

```python
pos_indices = torch.arange(seq_len, device=input_ids.device)
```

当 `seq_len = 4` 时，`pos_indices` 是 `[0, 1, 2, 3]`，表示 4 个 Token 在当前输入序列中的位置。随后 `self.pos_emb(pos_indices)` 根据这些整数索引查找可学习的位置嵌入。

下面用一个独立示例观察形状：

```python
pos_emb = torch.nn.Embedding(num_embeddings=8, embedding_dim=4)
pos_indices = torch.arange(4)
pos_embeds = pos_emb(pos_indices)

print(pos_indices.shape)  # torch.Size([4])
print(pos_embeds.shape)   # torch.Size([4, 4])
```

这里 `arange` 生成的是位置编号，位置嵌入层才负责将每个编号映射为长度为 `4` 的向量。输入序列长度不能超过位置嵌入表支持的位置数量。

## 参考

- [PyTorch 官方文档：torch.arange](https://docs.pytorch.org/docs/stable/generated/torch.arange.html)
- [项目中的张量切片与上下文长度说明](tensor_slicing_context_size.md)
