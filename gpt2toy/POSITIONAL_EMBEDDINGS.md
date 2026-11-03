# Transformer 与 GPT-2 位置嵌入的作用

`x_emb = tok_embeds + pos_embeds` 把 token 的身份信息与位置信息合并到同一个向量中。`tok_embeds` 表示“这是什么 token”，`pos_embeds` 表示“它在当前输入序列的哪个位置”，相加后的表示作为后续 Transformer 层的输入。

原始 Transformer、GPT-1 和 GPT-2 都采用了将位置表示与 token 嵌入相加的做法。主要区别在于：原始 Transformer 的主方案使用固定的正弦和余弦位置编码，GPT-1 与 GPT-2 使用可训练的位置嵌入表。

## 代码与张量形状

当前实现位于 [gpt2toy.py](/F:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy.py:221)。模型先建立两张独立的嵌入表：

```python
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
```

两张表的列数都是隐藏维度 `D = emb_dim`，但查表依据不同：

| 嵌入表 | 权重形状 | 查表依据 | 每次取出的内容 |
| --- | --- | --- | --- |
| `tok_emb` | `[V, D]` | token ID | 对应 token 的向量 |
| `pos_emb` | `[C, D]` | 位置编号 | 对应位置的向量 |

其中，`V` 是词表大小，`C` 是位置嵌入表容量，也就是配置中的最大上下文长度。

[前向计算](/F:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy.py:241) 为：

```python
# input_ids: [B, T]
tok_embeds = self.tok_emb(input_ids)                  # [B, T, D]
pos_indices = torch.arange(seq_len, device=input_ids.device)
pos_embeds = self.pos_emb(pos_indices)               # [T, D]
x_emb = tok_embeds + pos_embeds                      # [B, T, D]
x = self.drop_emb(x_emb)
```

`B` 表示 batch 中的序列数，`T` 表示输入序列长度。`pos_indices` 的值为 `0, 1, ..., T-1`，形状是 `[T]`；位置编号经过查表后才得到 `[T, D]` 的位置向量。

相加时，PyTorch 将 `[T, D]` 按照 `[1, T, D]` 广播到 batch 维。每条序列的第 `t` 个位置使用同一个位置向量，但 token 向量由该位置的 token ID 决定。逐位置写成公式就是：

\[
x_{b,t}=E_{\text{token}}[\text{input\_ids}_{b,t}]+E_{\text{position}}[t]
\]

这里的位置是当前输入窗口内的位置。当前 toy 实现每次前向都从位置 `0` 开始编号。

## 相同 token 在不同位置的数值示例

假设 tokenizer 将输入切成三个 token：

```text
位置编号       0      1      2
token          我     喜欢   我
```

两个“我”的 token ID 相同，因此查出的 token 向量相同。为了演示，设隐藏维度为 3，并使用以下假设数值：

```text
“我”的 token 向量 = [0.5, 0.2, 0.8]

位置 0 的向量     = [0.1, 0.0, 0.2]
位置 2 的向量     = [0.3, 0.1, 0.0]

第一个“我”的输入 = [0.5, 0.2, 0.8] + [0.1, 0.0, 0.2]
                 = [0.6, 0.2, 1.0]

第二个“我”的输入 = [0.5, 0.2, 0.8] + [0.3, 0.1, 0.0]
                 = [0.8, 0.3, 0.8]
```

相加后的两个输入向量不同，使后续网络有机会区分同一个 token 在不同位置的出现。这些数值只是计算示例；向量的各个分量没有预先指定的自然语言含义。

## 自注意力为什么需要位置线索

考虑两组 token：

```text
我 喜欢 你
你 喜欢 我
```

两组输入包含相同的 token，但顺序改变了意思。仅有 token 嵌入时，同一个 token 在不同位置的初始向量相同。

没有位置机制和掩码的标准自注意力，根据输入向量的内容计算关联。某个向量存放在张量的第 0 行，并不等于数值计算自动获得了“这是第 0 个位置”这一特征。如果同时重排输入位置，其输出也会相应重排；这称为置换等变性。

加入位置向量后，后续生成 Query、Key 和 Value 的输入表示包含了位置线索。模型可以通过训练学习内容与位置之间的关系，例如关注某个先前位置，或利用两个 token 之间的距离。

GPT-2 的因果掩码还规定了每个位置只能关注自己和前面的 token。这已经提供了一部分顺序结构，因此不能把“去掉位置嵌入”理解为“GPT-2 完全没有顺序信息”。两者的作用分别是：

- **因果掩码**限制哪些位置之间可以传递信息，防止预测时看到未来 token。
- **位置嵌入**给输入向量加入显式的位置特征，供网络学习和使用。

原始 Transformer 对解码器掩码和位置编码的说明分别见 [论文第 3.2.3 节与第 3.5 节](https://arxiv.org/html/1706.03762v7#S3.SS5)。

## 原始 Transformer 论文的说明

2017 年的 *Attention Is All You Need* 在第 3.5 节 Positional Encoding 中说明：模型没有循环和卷积，因此为了利用序列顺序，需要：

> “inject some information about the relative or absolute position of the tokens”

即向模型注入 token 的相对位置或绝对位置信息。随后论文明确说明，将位置编码加到编码器与解码器堆栈底部的输入嵌入上；位置编码与 token 嵌入具有相同的维度，因而能够逐元素相加。[原论文第 3.5 节](https://arxiv.org/html/1706.03762v7#S3.SS5)

原论文的主方案使用不同频率的正弦和余弦函数：

\[
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

\[
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

其中，`pos` 是位置编号，`2i` 和 `2i+1` 是向量中的维度编号。这些位置编码由公式产生，不是训练得到的查表参数。作者希望这种结构有助于模型学习相对位置关系，并可能外推到训练中未见过的序列长度。论文也测试了可学习的位置嵌入，两种方案的实验结果近乎相同。[原论文第 3.5 节](https://arxiv.org/html/1706.03762v7#S3.SS5)

此外，原论文第 3.4 节会先将 token 嵌入乘以 `sqrt(d_model)`，再与位置编码相加。因而它与当前代码遵循相同的相加思路，但输入处理细节并非完全一致。[原论文第 3.4 节](https://arxiv.org/html/1706.03762v7#S3.SS4)

## GPT-1 与 GPT-2 的论文和实现

### GPT-1 明确给出了相加公式

2018 年的 GPT-1 论文 *Improving Language Understanding by Generative Pre-Training* 在第 3.1 节公式 2 中写出：

\[
h_0=UW_e+W_p
\]

其中，`UW_e` 对应输入 token 的嵌入，`W_p` 对应输入位置的嵌入，`h_0` 是送入第一个 Transformer Block 的初始表示。这个公式对应当前代码中的 `tok_embeds + pos_embeds`。[GPT-1 论文第 3.1 节](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=3)

GPT-1 论文第 4.1 节还明确说明，使用可学习的位置嵌入，替代原始 Transformer 的正弦位置编码。[GPT-1 论文第 4.1 节](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5)

### GPT-2 沿用架构并在源码中直接相加

2019 年的 GPT-2 论文 *Language Models are Unsupervised Multitask Learners* 在第 2.3 节 Model 中说明，模型大体沿用 OpenAI GPT 的架构，并列出 LayerNorm 位置、初始化和上下文长度等修改。这一节没有重新详细解释位置嵌入的动机。[GPT-2 论文第 2.3 节](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4)

GPT-2 官方 `src/model.py` 建立了可训练的 `wte` 和 `wpe`，并直接执行：

```python
h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))
```

`wte` 是 token 嵌入表，`wpe` 是位置嵌入表，`tf.gather` 根据编号取出对应行。因此，这一行与当前 toy 模型中的 `x_emb = tok_embeds + pos_embeds` 采用相同的合并方式。[GPT-2 官方源码](https://github.com/openai/gpt-2/blob/master/src/model.py#L147-L158)

官方实现中的 `positions_for` 还使用 `past_length`：在有缓存的增量生成中，位置编号会接在已有 token 后面。当前 toy 前向使用 `torch.arange(seq_len)`，表示传入窗口内从零开始的局部位置。[GPT-2 位置编号源码](https://github.com/openai/gpt-2/blob/master/src/model.py#L141-L144)

| 实现 | 位置表示的来源 | 与 token 嵌入的合并方式 |
| --- | --- | --- |
| 原始 Transformer 主方案 | 固定的正弦和余弦公式 | token 嵌入经缩放后与位置编码相加 |
| GPT-1 | 可训练的位置嵌入表 | 逐元素相加 |
| GPT-2 | 可训练的绝对位置嵌入表 | 逐元素相加 |
| 当前 toy 模型 | `nn.Embedding(context_length, emb_dim)` | 通过广播逐元素相加 |

## 为什么选择相加

token 向量和位置向量都具有 `D` 个分量，相加后仍然是 `D` 维。这样既能把位置特征加入输入，也能直接保持后续 Transformer 层所需要的隐藏维度。如果直接拼接两个 `D` 维向量，结果会变为 `2D` 维，需要再调整后续计算或增加投影。

拼接也是可行的设计。例如，先拼接再线性投影回 `D` 维，采用行向量记法可以写成：

\[
x=[e;p]W=eA+pB
\]

其中，`e`、`p` 分别是 token 和位置向量，`A`、`B` 分别是作用于两部分的投影矩阵。它比直接使用 `x = e + p` 多了可学习的融合参数。若把拼接后的 `2D` 维作为后续网络的隐藏维度，则会增大投影和前馈网络的计算。相加是一种保持维度的设计选择，并不意味着拼接一定更差。

相加会混合两类信息，并不保证能从结果中唯一还原两个原始向量。例如：

```text
[1, 2] + [3, 4] = [4, 6]
[2, 3] + [2, 3] = [4, 6]
```

两个不同的组合可以得到相同结果，因此不存在任意输入下的无损分离保证。但相加也不意味着实际使用的有限组合一定发生冲突。GPT-2 的两张嵌入表与后续网络共同训练，目标是形成有助于预测的混合表示，而不是还原融合前的两份向量。

模型没有将前一半维度规定为内容、后一半维度规定为位置；两种信息可以分布在共同的维度中。

当前 toy 模型的 token 嵌入和位置嵌入均为随机初始化，不加载预训练权重。建立位置嵌入表提供了表达位置的能力；位置特征如何帮助语言预测，需要通过训练学习。

## 相加后，是否需要取回单独的两份嵌入

对当前 toy 模型和下表中检查的实现，后续 Transformer 都直接处理混合表示，没有先把它拆回 token 嵌入、位置嵌入的计算步骤。这里讨论的是一次前向中的特征计算，不表示嵌入权重表会被删除，或下一次前向不再查表。

本地 [gpt2small.py](/F:/LLM/BuildLLM/LLMs-from-scratch/gpt2/gpt2small.py:203) 的完整前向路径如下：

```python
tok_embeds = self.tok_emb(in_idx)
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
x = tok_embeds + pos_embeds
x = self.drop_emb(x)
for block in self.trf_blocks:
    x = block(x)
x = self.final_norm(x)
logits = self.out_head(x)
```

相加后，`tok_embeds`、`pos_embeds` 没有再被单独引用。注意力中的 Q/K/V 从当前隐藏状态投影得到，残差连接传递的也是当前混合表示。

### 开源实现的对照结果

以下是 2026-11-03 对本地文件及链接源码的静态检查结果；外部仓库的默认分支可能继续变化。

| 实现 | 嵌入如何进入 Transformer | 是否需要拆回原始嵌入 | 参考源码 |
| --- | --- | --- | --- |
| 本地 GPT-2 教学实现 | `tok_embeds + pos_embeds` 后，经 Dropout、各个 Block、最终归一化和输出层 | 不需要；后续只使用当前 `x` | [gpt2small.py](/F:/LLM/BuildLLM/LLMs-from-scratch/gpt2/gpt2small.py:203) |
| OpenAI 官方 GPT-2 | `wte`、`wpe` 查表后相加为 `h`，各层继续更新 `h` | 不需要；输出层另有词嵌入权重共享 | [src/model.py](https://github.com/openai/gpt-2/blob/master/src/model.py) |
| nanoGPT | token 和位置向量相加后，经 Dropout、各个 Block、最终归一化 | 不需要；输出层另有词嵌入权重共享 | [model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) |
| Hugging Face GPT-2 | `inputs_embeds + position_embeds` 得到 `hidden_states`，后续从当前隐藏状态生成 Q/K/V | 不需要；掩码辅助函数仍可读取原输入张量的元数据，见下文 | [modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) |
| Google 官方 BERT | token、句段类型和位置三种嵌入相加，经归一化和 Dropout 后送入编码器 | 不需要拆回三份嵌入 | [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py) |

这项结论适用于这里检查的嵌入相加路径，不能直接推广为所有位置机制都只在输入处使用一次位置特征。

### 权重共享不等于取回本次输入的 token 嵌入

需要区分两个对象：

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| `tok_embeds` | `[B, T, D]` | 本次输入按 token ID 查表得到的张量 |
| `tok_emb.weight` / `wte` | `[V, D]` | 整张词表的可学习权重 |

官方 GPT-2 在输出层执行：

```python
logits = tf.matmul(h_flat, wte, transpose_b=True)
```

它复用整张词嵌入表，为所有候选 token 计算分数，没有从隐藏状态中恢复本次输入的 `tok_embeds`。nanoGPT 也绑定了 `wte.weight` 和 `lm_head.weight`；Hugging Face GPT-2 的语言模型头同样声明了这项共享。[官方 GPT-2](https://github.com/openai/gpt-2/blob/master/src/model.py)、[nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)、[Hugging Face GPT-2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

本地 `gpt2small.py` 和 `gpt2toy.py` 创建了独立的 `out_head`，没有将其权重绑定到词嵌入表。

### 展示数据和辅助元数据仍可能被读取

当前 toy 模型在相加之后，把原始 `tok_embeds`、`pos_embeds` 和融合结果传给 `trace("embeddings", ...)`。[gpt2toy_diagnostics.py](/F:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy_diagnostics.py:264) 读取这些张量，用于打印、展示相加过程和保存可视化数据。这是直接读取仍然存在的张量，不是从 `x_emb` 中反推。

检查时的 Hugging Face GPT-2 还会将 `inputs_embeds` 传给掩码构造函数。该辅助函数使用其 batch 大小、序列长度、dtype 和 device 等元数据，不利用原始词元特征数值来分离嵌入。因此，“后续不需要还原两份嵌入”不等于“所有实现都不再引用原来的变量”。[GPT-2 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)、[掩码实现](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py)

## Q、K 如何利用相加后的信息

下面使用手工构造的数值，只计算一个头，并省略 LayerNorm、Dropout 和偏置。采用行向量记法：

\[
x_i=e_i+p_i,\qquad q_i=x_iW_Q,\qquad k_j=x_jW_K
\]

点积可以展开成四项：

\[
\begin{aligned}
q_i\cdot k_j={}&(e_iW_Q)\cdot(e_jW_K)\\
&+(e_iW_Q)\cdot(p_jW_K)\\
&+(p_iW_Q)\cdot(e_jW_K)\\
&+(p_iW_Q)\cdot(p_jW_K)
\end{aligned}
\]

它们分别对应内容与内容、内容与位置、位置与内容、位置与位置的交互。代码直接计算点积，不需要显式拆出这四项。

设输入为三个 token“猫、狗、猫”，嵌入如下：

```text
e_猫 = [1, 0, 0, 0]       p_0 = [0, 0, 1, 0]
e_狗 = [0, 1, 0, 0]       p_1 = [0, 0, 0, 1]
                         p_2 = [0, 0, 1, 1]
```

这里故意把内容和位置安排在不同维度，方便手算；真实模型没有这种固定分工。设一个头的线性投影为：

```text
对 x = [a, b, c, d]：
Q(x) = [a + c, b + d]
K(x) = [a + c, b + 2d]
```

| 位置与 token | 相加后的 x | Q | K |
| --- | --- | --- | --- |
| 0：猫 | `[1, 0, 1, 0]` | `[2, 0]` | `[2, 0]` |
| 1：狗 | `[0, 1, 0, 1]` | `[0, 2]` | `[0, 3]` |
| 2：猫 | `[1, 0, 1, 1]` | `[2, 1]` | `[2, 2]` |

位置 2 的查询 `q_2 = [2, 1]` 与三个 Key 的点积分别是 `4、3、6`。该位置可以关注这三个位置，每头维度为 2，因此注意力权重为：

\[
\operatorname{softmax}\left(\frac{[4,3,6]}{\sqrt{2}}\right)
\approx[0.17837,0.08795,0.73368]
\]

两个“猫”使用相同的词嵌入，但其位置向量不同，最终得到不同的 Q/K 和匹配分数。这个例子说明位置可以通过混合表示影响注意力，不表示真实模型一定更关注同词或当前位置。[缩放点积注意力公式](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1)

当前 toy 模型的第一层实际先计算 `LN(Dropout(e + p))`，再投影成 Q/K/V。LayerNorm 不能分配到两个加数上，因此上面的四项展开仅对省略归一化等步骤的简化模型精确成立。后续层接收的则是已经经过上下文计算的隐藏状态，也不能再简单写成最初的 `e + p`。

## 能否只给位置一个标量或别名

整数 `0, 1, 2, ...` 已经能够标识不同位置，但“能唯一标识”和“方便后续网络使用”不是同一件事。token ID 也是整数，模型仍然将其映射为词向量；位置索引经过查表得到向量，同样是为了参与后续特征计算。

位置的作用也不限于区分重复 token。“猫 追 狗”和“狗 追 猫”没有重复 token，词序仍然影响角色关系。向量位置表示为学习这些顺序和位置关系提供了特征；可学习的绝对位置向量并不天然保证向量距离等于位置距离。

### 作为额外一维拼接：可以设计

例如，将词向量 `[a, b, c, d]` 和位置标量 `i` 拼成 `[a, b, c, d, i]`，再投影到目标维度，是一种可行设计。在第一次线性投影中，位置的贡献具有如下形式：

\[
[e;i]W=eA+i\,b
\]

位置改变时，其贡献沿固定方向 `b` 变化。完整的位置向量则可以在多个方向上提供不同的位置特征。标量能够编码位置，但网络如何利用它，仍需结合架构和训练来判断。

### 广播加到每一维：会被 LayerNorm 消去统一平移

若直接使用 `x_i = e + i`，广播就会给每个分量加上同一个标量：

```text
e     = [1, 3, 2, 6]    均值 3
e + 1 = [2, 4, 3, 7]    均值 4
e + 2 = [3, 5, 4, 8]    均值 5
```

三者减去各自均值后均为 `[-2, 0, -1, 3]`，方差也相同。因此，从 LayerNorm 的公式可直接得到：

\[
\mathrm{LN}(e+i\mathbf{1})=\mathrm{LN}(e)
\]

对于当前 toy 模型，在推理模式关闭 Dropout 时，这种统一的位置偏移会在第一层注意力之前被归一化消去，无法让第一层 Q/K/V 区分同一 token 所处的位置。这个结论针对“同一个标量加到所有维度”；不适用于一般的非恒定位置向量。因果掩码仍然提供顺序结构，不能据此说模型完全失去顺序信息。[LayerNorm 公式](https://arxiv.org/html/1607.06450v1)

## 随机初始化不等于任意替换都不影响结果

初始化时不需要人工指定每个位置应对应哪个向量，但在固定其他模型参数后，位置权重的具体数值会影响输出。需要区分三个阶段：

- **未训练的 toy 模型**：随机位置权重已经参与数值计算，会影响输出；尚不能把这些影响解释为学会了语言中的位置关系。
- **训练过程**：词嵌入、位置嵌入、Q/K/V 等参数共同调整，形成对预测有用的配合。
- **训练完成后**：其他参数已经适应位置表的编码，随意替换位置权重会改变计算，不能期待保留原有行为。

也可以设计成随机生成位置表后将其固定，只训练其他参数。但网络必须学习如何使用这套固定编码；“可固定随机初始化”不等于“训练后可随意换表而行为不变”。

### 固定其他权重，只修改位置表的实验

2026-11-03 使用当前 `gpt2toy.GPTModel` 在 CPU 上新建小模型进行比较：随机种子为 `123`，词表大小 `6`，上下文长度 `5`，隐藏维度 `4`，注意力头数 `2`，层数 `2`，Dropout 为 `0`，`qkv_bias=False`，输入为 `[[2, 1, 3, 4]]`。实验没有训练模型，不是语言质量评测。

| 位置表处理方式 | 相对原始模型，所有位置和词表项中的最大 logits 绝对变化 | 末尾位置预测的下一个 token ID |
| --- | --- | --- |
| 原始随机权重 | `0` | `1` |
| 换成另一组随机权重 | `1.277707` | `5` |
| 全部置零 | `1.083382` | `5` |
| 将位置表的全部行反转 | `1.357375` | `5` |

在 `gpt2toy` 目录运行以下代码可复现比较；随机数实现或数值环境不同可能影响具体小数，表中的结论来自本次运行：

```python
import torch
from gpt2toy import GPTModel

torch.manual_seed(123)
cfg = dict(vocab_size=6, context_length=5, emb_dim=4, n_heads=2,
           n_layers=2, drop_rate=0.0, qkv_bias=False)
model = GPTModel(cfg).eval()
ids = torch.tensor([[2, 1, 3, 4]])

with torch.no_grad():
    original = model.pos_emb.weight.clone()
    base = model(ids)
    print("original", 0.0, base[0, -1].argmax().item())
    variants = [
        ("new_random", torch.randn_like(original)),
        ("zero", torch.zeros_like(original)),
        ("positions_reversed", original.flip(0)),
    ]
    for name, replacement in variants:
        model.pos_emb.weight.copy_(replacement)
        changed = model(ids)
        delta = (changed - base).abs().max().item()
        print(name, round(delta, 6), changed[0, -1].argmax().item())
    model.pos_emb.weight.copy_(original)
```

即使某次替换后 `argmax` 恰好仍然选中同一个 token，也不能推出“位置权重没有影响”：logits 和概率分布仍然可能发生变化。
