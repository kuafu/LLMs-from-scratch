# 多头潜在注意力（Multi-Head Latent Attention，MLA）

本扩展材料展示了使用多头潜在注意力（MLA）相较于常规多头注意力（MHA）能够节省多少内存。

&nbsp;
## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA），它是提高 MHA 计算效率的一种方法。消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，在 LLM 建模性能方面，GQA 与标准 MHA 的表现相当。

现在来看看多头潜在注意力（MLA）。DeepSeek V2、V3 和 R1 使用了 MLA（[论文](https://arxiv.org/abs/2412.19437)）。它采用了另一种节省内存的策略，而且特别适合与 KV 缓存搭配使用。与 GQA 共享键头和值头不同，MLA 会先将键张量和值张量压缩到低维空间，再存入 KV 缓存。

如下图所示，推理时会先将这些压缩张量投影回原来的大小，再使用它们。这会增加一次矩阵乘法，但能减少内存用量。

&nbsp;

![多头潜在注意力（MLA）](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺带一提，查询也会被压缩，但只在训练期间压缩，推理时不会。）

另外，如前所述，MLA 并非 DeepSeek V3 首创；它的前代模型 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 也使用了 MLA，并且是首次引入它的模型。V2 论文还包含一些有意思的消融研究，或许可以解释 DeepSeek 团队为何选择 MLA 而非 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="分组查询注意力（GQA）" width="500px" />

&nbsp;

如上图所示，GQA 的表现似乎不如 MHA，而 MLA 的建模性能优于 MHA，这可能就是 DeepSeek 团队选择 MLA 而非 GQA 的原因。（如果能看到 MLA 和 GQA 在“每个 Token 的 KV 缓存”方面的节省对比，也会很有意思！）

在继续介绍下一个架构组件之前，简单总结一下：MLA 是一种巧妙的方法，既能减少 KV 缓存的内存用量，在建模性能方面甚至还略胜 MHA 一筹。

&nbsp;
## MLA 的内存节省

节省的内存主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

相比之下，MHA 的 KV 缓存内存用量按以下公式计算：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着在 MLA 中，我们将 "embed_dim × 2 (K,V)" 减小为 "latent_dim"，因为如前图所示，我们只存储压缩后的潜在表示，而不存储完整的键向量和值向量。



你可以使用此文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本，为不同的模型配置应用该公式，查看使用 MLA 相较于 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

请注意，上面的压缩配置（`--emb_dim 2048 -> latent_dim 1024`）是为了达到与 GQA 类似的节省效果。实际使用时，压缩程度是一个需要仔细研究的超参数，因为 `latent_dim` 过小可能会损害建模性能（这与 GQA 中 `n_kv_groups` 取值过大的情况类似）。

下图进一步展示了在不同 `latent_dim` 值下，使用 MLA 相较于 MHA 节省的内存随上下文长度的变化：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="分组查询注意力（GQA）" width="500px" />

&nbsp;

你可以通过 `uv run plot_memory_estimates_mla.py` 复现该图。



&nbsp;
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了实践示例，用于在 GPT 模型实现中比较 MHA 和 MLA 的内存用量。

这里的 MLA 代码参考了 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 实现。

请注意，MLA 也可以与 [GQA](../04_gqa) 结合使用，但为简化起见，这里没有这样做。（目前我也不知道有哪些知名 LLM 采用了这种组合。）

还要注意，这个模型尚未训练，因此会生成无意义的文本。不过，你可以将它直接替换第 5 至第 7 章中的标准 GPT 模型，然后对其进行训练。

最后，这一实现使用了[另一个扩展章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

我们没有看到像上图中那么大的节省，原因有两点：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不只是注意力机制；模型中的全连接层占用了大部分内存（不过，这属于另一个分析主题）。
