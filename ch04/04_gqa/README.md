# 分组查询注意力（Grouped-Query Attention，GQA）

本扩展材料展示了使用分组查询注意力（GQA）相较于常规多头注意力（MHA）能够节省多少内存。

&nbsp;
## 简介

近年来，分组查询注意力（GQA）已成为多头注意力（MHA）的一种新标准替代方案，能提高计算和参数效率。需要说明的是，GQA 并非新方法，其历史可追溯到 2023 年的论文 [GQA：从多头检查点训练广义多查询 Transformer 模型](https://arxiv.org/abs/2305.13245)。甚至经典的 Llama 2 系列中的较大版本也使用了它。

简要来说，与每个注意力头都有自己的一组键和值的 MHA 不同，GQA 将多个注意力头分组，让它们共享相同的键和值投影，以降低内存用量。

例如，如下图所示，如果有 3 个键值组和 6 个注意力头，那么头 1 和头 2 共享一组键和值；头 3 和头 4、头 5 和头 6 则分别共享另外两组。

&nbsp;

![分组查询注意力（GQA）](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

共享键和值减少了键和值的总计算量，因此能够降低内存用量并提高效率。

总而言之，GQA 的核心思路是让多个查询头共享键和值头，从而减少键头和值头的数量。这样一来，(1) 模型的参数量会减少；(2) 推理时键张量和值张量所需的内存带宽会降低，因为需要存入 KV 缓存并从中读取的键和值更少。

虽然 GQA 主要是为提高 MHA 计算效率而采用的方法，但消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，在 LLM 建模性能方面，GQA 与标准 MHA 的表现相当。

不过，这一结论的前提是谨慎选择键值组的数量。如果所有注意力头都共享单个键值组，就会形成多查询注意力；这种极端情况下内存用量会进一步大幅降低，但建模性能可能受损。（另一个极端是键值组数量等于查询头数量，此时就回到了标准多头注意力。）

&nbsp;
## GQA 的内存节省

节省的内存主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

你可以使用此文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本，为不同的模型配置应用该公式，查看使用 GQA 相较于 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步展示了在不同键值组大小下，使用 GQA 相较于 MHA 节省的内存随上下文长度的变化：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="分组查询注意力（GQA）" width="500px" />

&nbsp;

你可以通过 `uv run plot_memory_estimates_gqa.py` 复现该图。

&nbsp;
## GQA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了实践示例，用于在 GPT 模型实现中比较 MHA 和 GQA 的内存用量。

请注意，[Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 的扩展材料中也使用了 GQA。不过，为了简化起见，此文件夹中的代码脚本修改的是传统上未使用 GQA 的 GPT 架构。

请注意，这个模型尚未训练，因此会生成无意义的文本。不过，你可以将它直接替换第 5 至第 7 章中的标准 GPT 模型，然后对其进行训练。

此外，这一实现使用了[另一个扩展章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们没有看到像上图中那么大的节省，原因有两点：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不只是注意力机制；模型中的全连接层占用了大部分内存（不过，这属于另一个分析主题）。
