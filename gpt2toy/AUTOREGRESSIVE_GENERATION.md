# 自回归生成：概率链式法则与连乘记号

> 本文是人工维护的概念说明，解释 GPT-2 Toy 如何从概率定义走到逐 token 生成。
> `gpt2toy_dissection.md` 是由导出器生成的单次运行数值报告；不要把本文件的手写内容直接加入其中，否则下次导出会覆盖它。

## 1. 问题：一段文本的概率如何表示？

把文本按 tokenizer 切分为 token 序列：

\[
x_1, x_2, \ldots, x_T
\]

其中，`x_t` 是第 `t` 个 token，`T` 是整个序列的 token 数。整段序列恰好为这组 token 的联合概率写作：

\[
P(x_1, x_2, \ldots, x_T)
\]

概率的链式法则允许把这个整体概率拆为逐步的条件概率：

\[
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1})
\]

这不是 GPT 特有的公式，而是概率论中的**概率链式法则**（chain rule of probability，也称联合概率的链式分解）。GPT 的工作是用带参数 `θ` 的神经网络来估计右侧的每一项。

## 2. 公式中每个符号的含义

| 记号 | 含义 |
| --- | --- |
| `x_t` | 第 `t` 个 token，例如一句话中的第 3 个词元。 |
| `x_1, ..., x_T` | 从第一个到第 `T` 个 token 组成的完整序列。 |
| `P(x_1, ..., x_T)` | 整个序列同时出现的联合概率。 |
| `P(x_t \mid x_1, ..., x_{t-1})` | 已知前面所有 token 时，第 `t` 个 token 是 `x_t` 的条件概率。竖线 `\mid` 读作“在……条件下”。 |
| `\prod_{t=1}^{T}` | 从 `t=1` 连乘到 `T` 的所有项。 |

连乘符号 `\prod` 是国际数学文献中通用的标准记号，作用和求和符号 `\sum` 相似，只是 `\sum` 做加法、`\prod` 做乘法：

\[
\prod_{t=1}^{4} a_t = a_1 \times a_2 \times a_3 \times a_4
\]

它是广泛约定的数学符号，不是类似 ISO 编号的一项正式标准。`\prod` 的字形来自大写希腊字母 Pi；这里它表示“乘积运算”，不是圆周率 `\pi`。

常见简写是：

\[
x_{<t} = (x_1, \ldots, x_{t-1}), \qquad
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
\]

两种写法完全等价。

## 3. 一个三 token 的展开例子

令：

\[
x_1=\text{“我”}, \qquad x_2=\text{“爱”}, \qquad x_3=\text{“你”}
\]

则：

\[
P(\text{“我 爱 你”})
= P(\text{“我”})
\times P(\text{“爱”}\mid\text{“我”})
\times P(\text{“你”}\mid\text{“我 爱”})
\]

若三个因子分别为 `0.1`、`0.5`、`0.4`，整段序列的概率为：

\[
0.1 \times 0.5 \times 0.4 = 0.02
\]

第一个因子没有普通文本前缀，数学上可理解为以前缀为空集为条件。实际语言模型通常会使用起始标记（`<BOS>`）或用户给定的 prompt 作为初始上下文。

## 4. 为什么叫“自回归生成”？

- **自**：下一步的条件来自同一序列已经出现的 token；模型生成出的 token 也会成为后续的上下文。
- **回归**：术语来自时间序列——用一个变量自身的历史值预测它的下一项。这里并不表示输出必须是连续数值，也不表示一定使用均方误差。
- **生成**：重复“预测下一个 token → 选出一个 token → 追加到序列”这一过程，逐步构成文本。

因此，Transformer 即使不是 RNN，也可以是自回归模型：决定因素是它按此前缀预测下一个 token 的概率分解方式，而不是网络是否包含循环层。

## 5. 带 prompt 的语言模型形式

若 `c` 是用户给出的 prompt，`y_1, ..., y_N` 是模型随后生成的 token，则更贴近实际使用的写法是：

\[
P_\theta(y_{1:N} \mid c)
= \prod_{t=1}^{N} P_\theta(y_t \mid c, y_{<t})
\]

其中 `θ` 表示模型参数。每个因子给出一个“下一 token 的概率分布”；它还没有决定唯一的输出 token。贪婪解码、随机采样、top-k 和 top-p 等策略负责从该分布中选择 token。

## 6. 它在当前 GPT-2 Toy 中怎样执行？

[`gpt2toy.py`](gpt2toy.py) 的 `generate_text_simple` 体现了这一个循环：

1. `idx_cond = idx[:, -context_size:]`：从当前已有序列取可用上下文窗口。
2. `model(idx_cond)`：得到每个位置对词表的 logits；`logits[:, -1, :]` 是“基于当前前缀预测下一个 token”的分数。
3. `idx_next = torch.argmax(logits_last, ...)`：当前实现采用**贪婪解码**，选择分数最高的 token。
4. `idx = torch.cat((idx, idx_next), dim=1)`：将刚选出的 token 追加回序列，使它成为下一轮的前缀一部分。

代码没有先显式计算 Softmax，是因为 Softmax 保持大小顺序：

\[
\operatorname*{argmax}_i z_i
= \operatorname*{argmax}_i \operatorname{softmax}(z)_i
\]

只有在需要查看或随机采样“概率分布”时，才必须显式计算 Softmax。

## 7. 训练和生成的前缀来源不同

训练时通常使用真实训练文本的前缀来预测下一个真实 token（teacher forcing），典型目标是最小化负对数似然：

\[
\mathcal{L}
= -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
\]

实际生成时，第一轮使用 prompt；从第二轮开始，前缀会逐渐包含模型自己生成的 token。这正是“自回归”在推理阶段最直观的表现。

## 8. 一句话总结

自回归语言模型不一次性决定整段文本，而是根据已有前缀不断估计下一个 token 的条件概率，再把选中的 token 写回前缀；`\prod` 则是将这些逐步条件概率组合为整段文本联合概率的通用连乘记号。
