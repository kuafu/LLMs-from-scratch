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

相加会混合两类信息，并不保证能从结果中唯一还原两个原始向量。它也不是将前一半维度规定为内容、后一半维度规定为位置。后续网络在这种共同包含内容与位置线索的表示上训练，学习对预测任务有用的特征。

当前 toy 模型的 token 嵌入和位置嵌入均为随机初始化，不加载预训练权重。建立位置嵌入表提供了表达位置的能力；位置特征如何帮助语言预测，需要通过训练学习。
