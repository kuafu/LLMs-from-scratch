# DeepSeek 稀疏注意力（DeepSeek Sparse Attention，DSA）

本扩展材料实现了 DeepSeek 稀疏注意力（DSA）机制。该机制在 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入，并首次发表于实验版本 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)。

以下概述参考了文章[《从 DeepSeek V3 到 V3.2：架构、稀疏注意力与 RL 更新》](https://magazine.sebastianraschka.com/p/technical-deepseek)中对 DSA 的讨论。

&nbsp;
## 简介

标准因果自注意力会让每个查询关注此前的所有 token，因此计算量为 O(L²)，KV 缓存则随序列长度 L 按 O(L) 增长。

[滑动窗口注意力（SWA）](../06_swa)已经展示了将注意力限制在固定局部窗口内可以大幅降低这一开销。在 SWA 中，每个查询 token 只关注附近一段局部范围内的先前 token。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="滑动窗口注意力" width="800px" />

*图 1. 滑动窗口注意力将每个查询 token 限制在固定的局部上下文窗口内。*

&nbsp;

DSA 采用了类似的思路，只关注一部分先前的 token。不过，它用可学习的选择机制替代了固定窗口。对于每个查询 token，模型会为候选的过去 token 评分，并只保留相关性最高的 token。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek 稀疏注意力的所选 token 模式" width="800px" />

*图 2. DeepSeek 稀疏注意力为每个查询 token 选择一组可学习的过去 token。*

&nbsp;

### 架构概述

DSA 在标准注意力的基础上增加了两个组件。

**1. 闪电索引器（Lightning Indexer）**

对于每个查询 token $t$ 和每个候选的过去 token $s$，索引器都会计算一个标量相关性分数。此实现明确写出了参考代码中的缩放因子：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量索引头的数量，
- $q_{t,j}$ 是 token $t$ 和头 $j$ 对应的索引器查询向量，
- $k_s$ 是过去 token $s$ 对应的共享索引器键向量，
- $w_{t,j}$ 是按头学习的门，并按 $1 / \sqrt{H_I}$ 缩放。

ReLU 会将负的点积贡献置为零，而经过门控的求和会汇总各个索引头的结果，为每个过去的 token 生成一个相关性分数。

在完整的 DeepSeek 模型中，索引器使用多头潜在注意力（MLA）压缩后的 token 表示。为了让 GPT 实现更简单，此文件夹直接从常规隐藏状态计算索引器的查询和键。

**2. Token 选择器（Token Selector）**

计算完所有索引器分数后，只保留分数最高的 top-K 位置。其余位置会在标准 softmax *之前*被屏蔽为 −∞，因此模型实际上只会关注 $k \ll L$ 个 token。

最终的稀疏性并非来自索引器中的 ReLU。由于分数是对多个索引头求和得到的，大多数最终分数仍可能非零。Token 选择器通过只保留 top-K 位置来形成稀疏模式。

在融合式生产实现中，这可以将注意力计算量从 O(L²) 降至 O(L·k)。此处的实现仍保留标准的稠密注意力分数矩阵，并在 softmax 之前应用 DSA 选出的 top-K 掩码。这样便于检查选择逻辑，但无法获得融合内核带来的计算节省。

下图总结了整个流程：闪电索引器为候选 token 评分，选择器保留 top-K 位置，生成的掩码则限制常规注意力 softmax 的范围。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek 稀疏注意力流程图" width="700px" />

*图 3. DSA 先为候选 token 评分，再保留 top-K 个 token 作为最终注意力掩码。*

&nbsp;
## 实现

`gpt_with_kv_dsa.py` 提供：

| 类 | 描述 |
|---|---|
| `LightningIndexer` | 用于评估过去 token 相关性的轻量多头评分器。 |
| `MultiHeadAttentionWithDSA` | 带有 DSA 稀疏掩码和可选 KV 缓存的标准 MHA。 |
| `GPTModel` | 将 `MultiHeadAttentionWithDSA` 替换进来的 GPT 风格模型。 |

此实现沿用了仓库中其他扩展材料的风格，可以作为独立脚本运行。它旨在让读者能在小型 GPT 风格模型中检查 DSA 机制。它没有实现 DeepSeek 完整的 MLA 体系、融合稀疏内核或针对部署的优化。

&nbsp;
## 使用方法

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

主要参数：

| 参数 | 默认值 | 描述 |
|---|---|---|
| `--index_n_heads` | 4 | 轻量索引器头的数量（H_I）。 |
| `--index_head_dim` | 64 | 每个索引器头的维度。 |
| `--topk` | 64 | 每个查询关注的 token 数量（k）。对于较短的序列，该值上限为序列长度。 |

&nbsp;
## 与 DeepSeek V3.2 的关系

完整规模的 DeepSeek-V3.2 模型将多头潜在注意力（MLA，参见 [../05_mla](../05_mla)）与 DSA 配合使用，索引器查询取自共享的压缩潜在表示，而不是原始输入。DeepSeek-V3.2 与首次引入并测试 DSA 的 DeepSeek-V3.2-Exp 使用相同的架构。

这里复现了键选择的核心思路：一种计算开销较低的可学习点积评分器会在注意力 softmax 之前，将每个查询限制在相关性最高的 token 上。

下面报告的推理成本对比有助于理解 DSA 对长上下文部署的意义。节省效果取决于生产内核和服务基础设施，因此不要将该图视为此文件夹中教学实现的基准结果。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="DeepSeek 稀疏注意力的推理成本对比" width="800px" />

*图 4. DeepSeek 技术报告中给出的 DSA 在长上下文服务中的推理成本节省情况，摘自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考资料

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡和参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的《从 DeepSeek V3 到 V3.2：架构、稀疏注意力与 RL 更新》：https://magazine.sebastianraschka.com/p/technical-deepseek
