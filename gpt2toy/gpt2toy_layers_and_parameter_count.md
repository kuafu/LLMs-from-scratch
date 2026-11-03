# GPT 模型层数与参数量计算

模型介绍里的“12 层、32 层、80 层”，通常指堆叠的 Transformer Block 数量，对应 `gpt2toy` 的 `n_layers`。模型名称里的“7B、8B”表示参数总量，需要把各模块中独立权重和偏置所包含的标量逐项相加。

当前 `gpt2toy` 默认配置有 **1 个 Transformer Block、308 个参数**。下面按当前源码拆解计算，并说明这些数字与 Hugging Face、LLaMA 配置的对应关系。

## 层数与模型结构参数

Hugging Face 是模型平台，不同架构的配置字段名称可能不同。当前配置定义在 [run_gpt2toy.py](run_gpt2toy.py) 的 `TOY_CONFIG` 中。

| 指标 | gpt2toy 当前配置 | Hugging Face GPT-2 字段 | Hugging Face LLaMA 字段 |
| --- | --- | --- | --- |
| Transformer Block 数量 | `n_layers = 1` | `n_layer` | `num_hidden_layers` |
| 每个 token 的隐藏维度 | `emb_dim = 4` | `n_embd` | `hidden_size` |
| 每层注意力头数 | `n_heads = 2` | `n_head` | `num_attention_heads` |
| FFN 中间维度 | `4 * emb_dim = 16`，代码固定 | `n_inner`，默认取隐藏维度的 4 倍 | `intermediate_size` |
| 最大上下文长度 | `context_length = 5` | `n_positions` | `max_position_embeddings` |
| 词表大小 | `vocab_size = 6` | `vocab_size` | `vocab_size` |

字段定义参见 Hugging Face 官方的 [GPT-2 配置说明](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config) 和 [LLaMA 配置说明](https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaConfig)。

[gpt2toy.py](gpt2toy.py) 根据层数创建模型块：

```python
self.trf_blocks = nn.ModuleList([
    TransformerBlock(cfg) for _ in range(cfg["n_layers"])
])
```

一个块内部包含多头注意力、FFN、两处 LayerNorm 和两条残差连接。通常整个块算一层，其中的每个 `Linear` 不再分别计入模型介绍的层数；输入 Embedding、最终 LayerNorm 和输出头通常也不计入。

每个块拥有独立参数。假设输入为 1 条文本、3 个 token，隐藏维度为 4，把 `n_layers` 改成 3，隐藏状态依次经过三个块，形状始终为 `[1, 3, 4]`，数值逐层更新。层数增加会增加参数量，隐藏维度保持为 4。

## 参数量的计数规则

一个参数指一个独立的权重或偏置标量。例如，形状为 `[6, 4]` 的权重矩阵包含 `6 * 4 = 24` 个参数。

PyTorch 的 `nn.Linear(in_features, out_features)` 保存：

- 权重 `weight`：形状 `[out_features, in_features]`，参数量为 `out_features * in_features`。
- 偏置 `bias`：默认存在，形状 `[out_features]`；设置 `bias=False` 时不计此项。

因此，`Linear(4, 16)` 默认有 `16 * 4 + 16 = 80` 个参数。这里矩阵形状按 PyTorch 实际保存的方向书写。

本次计算采用以下配置：

```python
V = 6    # vocab_size，词表大小
C = 5    # context_length，位置嵌入表容量
D = 4    # emb_dim，隐藏维度
N = 1    # n_layers，Transformer Block 数量
H = 2    # n_heads，每层注意力头数
F = 16   # FFN 中间维度，当前代码取 4 * D
```

另外，`qkv_bias=False`，但注意力输出投影和 FFN 的线性层仍然带有 bias。词元 Embedding 与模型输出层使用两份独立权重。

## 当前模型的逐项计算

### Embedding 共 44 个参数

当前模型使用词元嵌入和可学习的绝对位置嵌入。

| 模块 | 权重形状 | 计算 | 参数量 |
| --- | --- | --- | ---: |
| `tok_emb` 词元嵌入 | `[V, D] = [6, 4]` | `6 * 4` | 24 |
| `pos_emb` 位置嵌入 | `[C, D] = [5, 4]` | `5 * 4` | 20 |
| 合计 | | `24 + 20` | **44** |

一次输入即使只用了 3 个 token，仍然要统计完整嵌入表的参数。token 数决定本次查表得到多少个向量，嵌入表本身的大小由配置决定。

### 每个块的注意力共 68 个参数

`MultiHeadAttention` 包含 Q、K、V 三个投影和一个输出投影。Q、K、V 没有 bias，输出投影有 bias。

| 模块 | 权重形状 | bias 形状 | 计算 | 参数量 |
| --- | --- | --- | --- | ---: |
| `W_query` | `[4, 4]` | 无 | `4 * 4` | 16 |
| `W_key` | `[4, 4]` | 无 | `4 * 4` | 16 |
| `W_value` | `[4, 4]` | 无 | `4 * 4` | 16 |
| `out_proj` | `[4, 4]` | `[4]` | `4 * 4 + 4` | 20 |
| 合计 | | | `16 + 16 + 16 + 20` | **68** |

这里不能再乘一次 `n_heads=2`。Q、K、V 投影各自已经生成完整的 4 维表示，随后通过 `view` 和 `transpose` 拆成两个头，每个头 2 维。拆分操作没有创建新的权重。

在当前普通多头注意力实现中，固定 `D=4`，把头数从 2 改为 1，注意力参数量仍是 68。`Q @ K.T`、Softmax 和计算得到的 `attn_weights` 都不引入新的可学习参数。

### 每个块的 FFN 共 148 个参数

FFN 的结构是 `4 -> 16 -> 4`，两个线性层都带有 bias，中间的 GELU 没有可学习参数。

| 模块 | 权重形状 | bias 形状 | 计算 | 参数量 |
| --- | --- | --- | --- | ---: |
| 升维层 `ff.layers[0]` | `[16, 4]` | `[16]` | `16 * 4 + 16` | 80 |
| 降维层 `ff.layers[2]` | `[4, 16]` | `[4]` | `4 * 16 + 4` | 68 |
| 合计 | | | `80 + 68` | **148** |

FFN 对每个 token 使用同一组参数，因此也不需要乘输入 token 数或 batch size。

### 所有 LayerNorm 共 24 个参数

当前自定义 `LayerNorm` 包含两个可学习向量：`scale` 和 `shift`。两者形状都是 `[D] = [4]`，所以每处 LayerNorm 有 `4 + 4 = 8` 个参数。

| 位置 | 数量 | 计算 | 参数量 |
| --- | ---: | --- | ---: |
| 每个块内的 `norm1` 和 `norm2` | 2 | `2 * 8` | 16 |
| 块外的 `final_norm` | 1 | `1 * 8` | 8 |
| 当前模型合计 | 3 | `16 + 8` | **24** |

均值和方差由输入计算得到；`eps` 是固定常数，都不计入参数量。

### 输出层共 24 个参数

输出层为 `nn.Linear(D, V, bias=False)`，将每个 token 的 4 维表示映射为 6 个词表得分。

```text
out_head.weight 的形状 = [V, D] = [6, 4]
输出层参数量           = 6 * 4 = 24
```

虽然形状与 `tok_emb.weight` 相同，当前代码创建的是两份独立权重，因此两处都需要计数。

### 总参数量为 308

| 部分 | 参数量 |
| --- | ---: |
| 词元与位置 Embedding | 44 |
| 一个块内的注意力 | 68 |
| 一个块内的 FFN | 148 |
| 一个块内的两处 LayerNorm | 16 |
| 最终 LayerNorm | 8 |
| 输出层 | 24 |
| 总计 | **308** |

一个 Transformer Block 有 `68 + 148 + 16 = 232` 个参数，块外共有 `44 + 8 + 24 = 76` 个参数。因此，当前模型也可以写为：

```text
总参数量 = 块外参数 + 层数 * 每个块的参数
         = 76 + 1 * 232
         = 308
```

## 当前结构的总参数公式

沿用前面的符号，当前结构满足：普通多头注意力、QKV 不带 bias、FFN 中间维度为 `4D`、词元嵌入与输出层不共享权重。

```text
词元 Embedding        = VD
位置 Embedding        = CD
每个块的注意力        = 3D² + D² + D = 4D² + D
每个块的 FFN          = (4D² + 4D) + (4D² + D) = 8D² + 5D
每个块的两处 LayerNorm = 4D
最终 LayerNorm        = 2D
输出层                = VD
```

所以：

```text
每个块参数量 = 12D² + 10D

总参数量 P = 2VD + CD + N(12D² + 10D) + 2D
```

代入当前配置：

```text
P = 2 * 6 * 4 + 5 * 4 + 1 * (12 * 4² + 10 * 4) + 2 * 4
  = 48 + 20 + 232 + 8
  = 308
```

只调整一个条件时，可以得到以下结果：

| 调整 | 参数总量 | 原因 |
| --- | ---: | --- |
| 当前配置 | 308 | `76 + 232` |
| `n_layers = 2` | 540 | `76 + 2 * 232` |
| `n_layers = 3` | 772 | `76 + 3 * 232` |
| `n_heads = 1` | 308 | 隐藏维度不变，分头方式改变 |
| `qkv_bias = True` | 320 | 当前一个块新增 `3 * D = 12` 个 bias 参数 |
| 将输出层权重绑定到词元嵌入权重 | 284 | 两个模块共享同一个参数矩阵，总量减少 `VD = 24` |

最后一行是显式修改权重绑定关系后的结果，当前默认模型不共享这两份权重。一般情况下，启用 QKV bias 会额外增加 `3ND` 个参数。

固定其他条件时，参数量随层数 `N` 线性增长；每个块的主项为 `12D²`，所以隐藏维度 `D` 对参数量的影响更大。

## 7B 和 8B 的计算含义

`B` 是 billion，即十亿；`M` 是 million，即百万。

```text
7B = 7,000,000,000 个参数，约 70 亿
8B = 8,000,000,000 个参数，约 80 亿

以 B 为单位的参数量 = 独立参数标量总数 / 1,000,000,000
```

真实大模型也按相同原则，把嵌入、所有模型块、归一化和输出层的参数相加。模型名称中的数字可能经过取整，精确总量应以具体实现和权重为准。

上面的代数公式只适用于当前 toy 结构。对于采用 RoPE、SwiGLU 或分组查询注意力 GQA 的 LLaMA 模型，应按其实际模块和矩阵形状重新计算；相关架构说明参见 [Hugging Face LLaMA 文档](https://huggingface.co/docs/transformers/en/model_doc/llama)。

参数量和以下数字需要分开理解：

- 输入 token 数、batch size、生成次数会影响计算量或中间张量大小，不会为每次计算复制一套模型参数。
- 因果掩码在当前模型中是一个 `[5, 5]` 的 buffer，包含 25 个元素，但不是模型参数。直接把 `state_dict()` 中所有张量的元素相加，会把这类 buffer 也算进去。
- 冻结权重会减少可训练参数量，模型总参数量保持不变。
- FP32、FP16、INT8 等表示精度影响每个权重占用的存储空间，参数数量本身保持不变。

## 用 PyTorch 复核

在 `gpt2toy` 目录中运行以下代码，可以直接列出每个参数张量的名称、形状和元素数量。导入 `run_gpt2toy` 只读取配置，不会执行完整演示或重写演示报告。

```python
from gpt2toy import GPTModel
from run_gpt2toy import TOY_CONFIG

model = GPTModel(dict(TOY_CONFIG))

for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape), parameter.numel())

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("总参数量:", total)          # 308
print("可训练参数量:", trainable)  # 308
assert total == 308
```

`numel()` 返回张量中的标量数量，`model.parameters()` 默认会去除重复的 Parameter 引用，因此同一个共享权重不会被重复统计。

仓库已有的参数统计实现位于 [gpt2toy_diagnostics.py](gpt2toy_diagnostics.py) 的 `collect_and_print_model_info`，也使用 `sum(p.numel() for p in model.parameters())` 计算总参数量。
