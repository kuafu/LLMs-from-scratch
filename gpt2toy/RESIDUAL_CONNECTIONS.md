# Transformer 的残差连接：两次更新与源码对照

当前 GPT-2 Toy 的每个 `TransformerBlock` 都执行两次残差更新：Attention 后一次，FFN 后一次。GPT-2、Llama，以及 Hugging Face Transformers 中对应的标准实现，也采用这一顺序结构。

本文是人工维护的技术说明，对应 [gpt2toy.py](gpt2toy.py) 中的 `TransformerBlock.forward`。运行产生的具体数值可查看 `gpt2toy.txt`；[gpt2toy_dissection.md](gpt2toy_dissection.md) 是自动生成报告。

## 1. 什么是残差连接？

残差连接（Residual Connection）的基本计算为：

\[
y = x + F(x)
\]

- `x`：通过恒等旁路（Identity Shortcut）直接传递的输入。
- `F(x)`：经过子层计算得到的更新量，在本文的 Transformer 中包含归一化、Attention 或 FFN，以及相应的 Dropout。
- `y`：输入与更新量逐元素相加后的结果。

若希望整个模块表示映射 `H(x)`，可以把分支学习的函数写为 `F(x) = H(x) - x`，因而 `H(x) = x + F(x)`。“残差”在这里指相对于输入的差值，不要求更新量很小，也不等同于训练损失。残差学习的这一表达可参见 [ResNet 原论文](https://arxiv.org/abs/1512.03385)。

如果分支输出为零，模块就直接传递 `x`。恒等旁路为深层网络提供了直接传递表示和梯度的路径，有助于优化。对于可微的 `F`，展平张量后的雅可比矩阵满足：

\[
\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}
\]

其中 `I` 来自恒等旁路。这不保证所有深层网络的梯度都不会消失或爆炸。

## 2. 为什么一个 Block 中有两次更新？

当前 Block 有两个依次执行的主要子层：因果自注意力（Attention）和逐位置前馈网络（FFN）。每个子层都将自己的输出加回残差流，因此有两次更新。

令 `x0` 为 Block 输入，`x1` 为第一次更新结果，`x2` 为 Block 输出：

\[
x_1 = x_0 + \operatorname{Dropout}(\operatorname{Attention}(\operatorname{LN}_1(x_0)))
\]

\[
x_2 = x_1 + \operatorname{Dropout}(\operatorname{FFN}(\operatorname{LN}_2(x_1)))
\]

这里的 `LN1` 与 `LN2` 是两处独立的 LayerNorm。归一化在子层之前执行，这种结构称为前置归一化（Pre-LN）。

**第二次更新的旁路和 FFN 输入都基于 `x1`。** 不能把它改写成下面这种由同一个 `x0` 分别计算 Attention 和 FFN 的形式：

```python
# 与当前顺序结构一般不等价：FFN 没有使用 Attention 更新后的表示。
x2 = x0 + Attention(LN1(x0)) + FFN(LN2(x0))
```

Attention 负责聚合当前 token 可以看到的上下文；FFN 对每个位置的特征进行变换。FFN 自身不在 token 之间交换信息，但它接收的 `x1` 已经包含 Attention 聚合的信息。

## 3. 与当前代码逐项对应

省略教学 trace 回调和断言后，当前 `TransformerBlock.forward` 的计算为：

```python
# 第一次残差更新
shortcut1 = x
x_norm1 = self.norm1(x)
att_raw = self.att(x_norm1)
att_out = self.drop_shortcut(att_raw)
x_after_res1 = shortcut1 + att_out

# 第二次残差更新
x = x_after_res1
shortcut2 = x
x_norm2 = self.norm2(x)
ff_raw = self.ff(x_norm2)
ff_out = self.drop_shortcut(ff_raw)
x_out = shortcut2 + ff_out

return x_out
```

| 数学记号 | 当前变量 | 含义 |
| --- | --- | --- |
| `x0` | 原始 `x`、`shortcut1` | Block 输入，也是第一次更新的旁路 |
| 第一次更新量 | `att_out` | Attention 输出经过残差分支 Dropout 后的结果 |
| `x1` | `x_after_res1`，之后的 `x`、`shortcut2` | 第一次更新结果，也是第二次更新的旁路 |
| 第二次更新量 | `ff_out` | FFN 输出经过残差分支 Dropout 后的结果 |
| `x2` | `x_out` | 两次更新完成后的 Block 输出 |

`shortcut1 = x`、`x = x_after_res1`、`shortcut2 = x` 都只是局部变量绑定，不会复制张量。`x = x_after_res1` 不会修改调用者传入的原始张量。当前残差相加使用 `+`，不执行原地写入，因此保存旁路引用是可行的。

### 形状与参数量

当前残差相加的两个操作数形状相同，都是 `[B, T, D]`。例如，默认首次生成前向输入有 4 个 token，残差流形状为 `[1, 4, 4]`。

FFN 中间会经历 `[B, T, D] -> [B, T, 4D] -> [B, T, D]`，但与旁路相加前已恢复为 `D` 维。这里做的是逐元素相加，结果仍为 `[B, T, D]`；拼接会改变维度，不是当前残差操作。

恒等旁路和相加本身没有可学习参数；Attention、FFN 和 LayerNorm 有自己的参数。一个 Block 内的两次残差更新也不会把 `n_layers` 算成 2；模型层数的约定参见 [GPT 模型层数与参数量计算](gpt2toy_layers_and_parameter_count.md)。

## 4. 一个可手算的例子

以下是为演示相加而假设的单个 token 的 4 维向量，不是当前随机模型的真实 Attention 或 FFN 输出：

```text
x0       = [ 1.0,  2.0,  3.0,  4.0]
att_out  = [ 0.1, -0.2,  0.3,  0.0]
x1       = [ 1.1,  1.8,  3.3,  4.0]  # x0 + att_out

ff_out   = [-0.1,  0.2,  0.0,  0.5]  # 假设由 LN2(x1) 经 FFN/Dropout 得到
x2       = [ 1.0,  2.0,  3.3,  4.5]  # x1 + ff_out
```

第二次相加的旁路是 `[1.1, 1.8, 3.3, 4.0]`。在真实模型中，`ff_out` 也必须用这个 `x1` 经过第二处归一化和 FFN 计算得到，不能仅把某个固定更新量机械地加到任意输入上。

手动核对当前运行日志时，可以依次检查：

1. 第 5 步：`shortcut1 + att_out` 是否逐元素等于 `x_after_res1`。
2. 第 6、7 步：`shortcut2` 和第二处 LayerNorm 的输入，是否都是第一次更新结果。
3. 第 10 步：`shortcut2 + ff_out` 是否逐元素等于 `x_out`。

逐行日志保留小数点后 8 位，展示值仍经过舍入，手算应允许末位误差。代码中的 `torch.equal` 检查的是张量实际存储值的逐元素精确相等，不是打印字符串相等。

## 5. Dropout 为零时发生了什么？

当前 [run_gpt2toy.py](run_gpt2toy.py) 设置 `drop_rate = 0.0`，并调用 `model.eval()`。以下任意一个条件都足以让 `nn.Dropout` 保持正常有限输入的数值不变：

- `p = 0`：不丢弃元素，不缩放数值。
- `training = False`：推理模式下直接传递输入，与配置的 `p` 大小无关。

因此当前可以简化为：

```python
att_out = att_raw
ff_out = ff_raw
```

当前源码在两处 Dropout 后加入了用于手工核对的断言：

```python
assert torch.equal(att_out, att_raw), "Dropout 改变了 att_raw 的数值"
assert torch.equal(ff_out, ff_raw), "Dropout 改变了 ff_raw 的数值"
```

这些断言检查的是当前教学配置的预期。若以后在训练模式启用 `0 < p < 1`，Dropout 会随机置零部分元素，并把保留元素乘以 `1 / (1 - p)`，此时不再保证输入输出相等；应根据训练用途调整或移除这些断言。Dropout 语义参见 [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)。

Dropout 不改变值，不代表整个 Attention 或 FFN 分支没有更新。当前仍然要把 `att_raw` 和 `ff_raw` 分别加回旁路，两次残差更新仍然存在。

## 6. GPT-2、Llama 与 Hugging Face 的源码对照

下表按 2026-11-03 查阅的源码说明。链接指向相应仓库分支，后续实现可能变化；Hugging Face Transformers 是实现多种架构的库，残差结构由具体模型决定。

| 模型及实现 | 对应函数或类 | 标准 Block 的残差更新 | 子层前归一化 |
| --- | --- | --- | --- |
| [OpenAI GPT-2](https://github.com/openai/gpt-2/blob/master/src/model.py#L111-L118) | `block` | Attention 后一次，MLP 后一次 | LayerNorm |
| [Hugging Face GPT-2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L246-L279) | `GPT2Block.forward` | 无交叉注意力时两次 | LayerNorm |
| [Meta 官方 Llama 2 实现](https://github.com/meta-llama/llama/blob/main/llama/model.py#L345-L369) | `TransformerBlock.forward` | Attention 后一次，FFN 后一次 | RMSNorm |
| [Hugging Face Llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L268-L285) | `LlamaDecoderLayer.forward` | Attention 后一次，MLP 后一次 | RMSNorm |
| [Hugging Face Qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L247-L264) | `Qwen2DecoderLayer.forward` | Attention 后一次，MLP 后一次 | RMSNorm |

这里的 MLP 对应本文的 FFN 子层。将归一化方式统一记作 `Norm`、省略 Dropout 等细节，可以用同一个骨架理解这些标准 Block：

```python
x1 = x0 + Attention(Norm1(x0))
x2 = x1 + FFN(Norm2(x1))
```

两次残差更新相同，并不表示子层内部计算相同。例如，GPT-2 使用 LayerNorm 和 GELU 前馈网络；上述 Meta Llama 实现使用 RMSNorm 和 SwiGLU 前馈网络。

Hugging Face Llama 的 `post_attention_layernorm` 位于第一次残差相加之后、MLP 之前。名字中的 `post_attention` 表示它在 Attention 阶段之后；相对于 FFN，它仍然是前置归一化，不能据此把整个结构判断为 Post-LN。

### 两次更新不是所有架构的固定规则

Hugging Face GPT-2 支持可选交叉注意力。当模型设置 `add_cross_attention=True`，并在前向中实际传入 `encoder_hidden_states` 执行交叉注意力时，会在自注意力和 FFN 之间增加一次残差更新，使该 Block 共有三次。仅配置开关为真、但未执行该分支时，仍为两次。具体条件和相加位置见 [GPT2Block 的交叉注意力分支](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L257-L279)。

对其他模型，应检查它的 Block 前向实现，分别确认旁路取自哪个张量、子层基于哪个输入计算，以及最终在哪里相加。
