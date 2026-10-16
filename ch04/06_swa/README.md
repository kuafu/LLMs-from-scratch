# 滑动窗口注意力（Sliding Window Attention，SWA）

本扩展材料展示了使用滑动窗口注意力（SWA）相较于常规多头注意力（MHA）能够节省多少内存。



&nbsp;
## 简介

什么是滑动窗口注意力（SWA）？常规自注意力是一种*全局*注意力机制，因为序列中的每个元素都可以访问其他所有元素。与之相对，SWA 可以看作*局部*注意力，因为它会限制当前查询位置周围的上下文范围。下图对此进行了说明。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="滑动窗口注意力（SWA）" width="500px" />

如上图所示，每个 token 不再关注所有先前的 token，而只关注其所在位置周围固定大小的局部窗口。这种局部注意力可以大幅缩小 KV 缓存。

在本节余下部分，我们将结合 [Gemma 3](https://arxiv.org/abs/2503.19786) 介绍 SWA；我们在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从零开始实现了 Gemma 3。

滑动窗口注意力最初由 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150)提出。我们之所以重点介绍 Google 的 Gemma 模型，是因为这些优秀的开放权重模型表明，滑动窗口注意力确实适用于近期的高性能模型。

[Gemma 2](https://arxiv.org/abs/2408.00118) 采用混合方案，以 1:1 的比例组合局部（滑动窗口）注意力层和全局注意力层。每个 token 可以关注由 4 k 个 token 组成的上下文窗口。采用这种 1:1 混合比例，是为了平衡效率和全局上下文建模，因为只使用局部注意力的 LLM 可能会受到过多限制。

[Gemma 3](https://arxiv.org/abs/2503.19786) 进一步朝着提高效率的方向改进了这一设计。它将滑动窗口注意力层与完整注意力层的比例设为 5:1，也就是每五个局部注意力层对应一个全局注意力层。此外，滑动窗口大小也从 Gemma 2 中的 4096 个 token 缩小到 Gemma 3 中的 1024 个 token。

有意思的是，Gemma 3 技术报告中的消融研究表明，这些变化对模型整体质量的影响很小。换句话说，通过滑动窗口注意力大幅节省内存和计算资源时，建模性能损失很小。



&nbsp;
## 滑动窗口注意力（SWA）的内存节省

节省的内存主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们将上面公式中的序列长度（seqlen）替换为窗口大小 W。因此，使用滑动窗口注意力时，KV 缓存大小会缩小到原来的 "W / seqlen"。（请注意，为简化起见，这里假设每一层都使用滑动窗口注意力。）


你可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，为不同的模型配置应用该公式，查看使用 SWA 相较于 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

请注意，Gemma 3 将 SWA 与 GQA 结合使用。

下图进一步展示了在不同上下文长度下，使用 SWA 相较于 MHA 节省的内存：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="滑动窗口注意力（SWA）" width="800px" />

&nbsp;

你可以通过以下方式复现这些图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了实践示例，用于在 GPT 模型实现中比较 MHA 和 SWA 的内存用量。

请注意，SWA 也可以与 MLA 和 GQA 结合使用（如前所述），但为简化起见，这里没有这样做。

请注意，这个模型尚未训练，因此会生成无意义的文本。不过，你可以将它直接替换第 5 至第 7 章中的标准 GPT 模型，然后对其进行训练。

此外，这一实现使用了[另一个扩展章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有看到像上图中那么大的节省，原因有两点：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不只是注意力机制；模型中的全连接层占用了大部分内存（不过，这属于另一个分析主题）。
