# 跨层 KV 共享（Cross-Layer KV Sharing）

本扩展材料展示了将跨层 KV 共享与 KV 缓存结合使用时能够节省多少内存。

&nbsp;
## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA），其中多个查询头共享同一组键头和值头。跨层 KV 共享则将类似思路应用于不同的 Transformer 层。

后续层不再各自计算新的键和值投影，而是复用较早层的 K/V 张量。它们仍会计算自己的查询，因此每层都可以形成自己的注意力模式。主要的内存节省来自缓存中存储的 K/V 张量数量减少。

这一思路也称为跨层注意力。Brandon *等人*在论文[《通过跨层注意力缩小 Transformer 键值缓存》](https://arxiv.org/abs/2405.12981)中对此进行了介绍。Gemma 4 E2B 和 E4B 使用了类似的共享 KV 缓存方案，因此本节是对本章 GQA、MLA 和 SWA 示例的有益补充。

&nbsp;

<img src="gemma4-kv-sharing.webp" alt="跨层 KV 共享" width="800px" />

&nbsp;

在 [Gemma 4](../../ch05/17_gemma4) 中，KV 共享与 GQA 或 MQA 以及滑动窗口注意力结合使用。在此文件夹的简化 GPT 示例中，我们只实现了跨层 KV 共享部分，因此代码仍专注于核心机制。

这里采用的简化规则如下：

1. 较早的层计算并缓存自己的 K/V 张量。
2. 后续层复用较早的某个生成层最近一次生成的 K/V 张量。
3. 所有层仍会计算自己的查询投影。

这样可以减少会随上下文长度增长的 K/V 缓存数量。代价是部分层不再拥有自己的 K/V 投影，因此模型容量会降低。

&nbsp;
## KV 共享的内存节省

常规 KV 缓存的内存用量按以下方式计算：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_layers x 2 (K,V) x bytes_per_elem

使用跨层 KV 共享时，我们将 `n_layers` 替换为生成 K/V 的层数：

bytes = batch_size x seqlen x head_dim x n_kv_heads x n_kv_producing_layers x 2 (K,V) x bytes_per_elem

你可以使用此文件夹中的 [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) 脚本，将该公式应用到不同的模型配置中：

```bash
# Gemma 4 E2B-like setup
uv run memory_estimator_kv_sharing.py \
  --context_length 131072 \
  --emb_dim 2048 \
  --n_heads 8 \
  --n_layers 35 \
  --n_kv_groups 8 \
  --n_kv_producing_layers 15 \
  --batch_size 1 \
  --dtype bf16

# Gemma 4 E4B-like setup
# uv run memory_estimator_kv_sharing.py \
#   --context_length 131072 \
#   --emb_dim 2560 \
#   --n_heads 8 \
#   --n_layers 42 \
#   --n_kv_groups 4 \
#   --n_kv_producing_layers 24 \
#   --batch_size 1 \
#   --dtype bf16

==== Config ====
context_length         : 131072
emb_dim                : 2048
n_heads                : 8
n_layers               : 35
n_kv_groups            : 8
n_kv_producing_layers  : 15
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 256
GQA n_kv_heads         : 1

==== KV-cache totals across all layers ====
MHA total KV cache        : 37.58 GB
GQA total KV cache        : 4.70 GB
MHA + KV sharing          : 16.11 GB
GQA + KV sharing          : 2.01 GB
Ratio (MHA / GQA+sharing) : 18.67x
Savings vs MHA            : 94.64%
```

这是一个类似 Gemma 4 E2B 的配置。35 层中有 15 层负责生成 K/V，其余层复用较早层的 K/V 张量。对于类似 E4B 的配置，相应数字是总共 42 层，其中 24 层负责生成 K/V。

下图展示了类似 E2B 和 E4B 的配置所节省的内存。为简化起见，这些图没有计入滑动窗口注意力带来的额外节省。

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="类似 Gemma 4 E2B 配置的 KV 共享内存节省" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="类似 Gemma 4 E4B 配置的 KV 共享内存节省" width="800px" />

&nbsp;

你可以通过以下命令复现类似的图：

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```

&nbsp;
## KV 共享代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 脚本提供了实践示例，用于比较常规 MHA 与跨层 KV 共享变体。

查看实现细节最简单的方法，是对比 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 的文件差异。两者的注释特意保持相似，这样差异对比就能突出 KV 共享方面的改动。

请注意，这个模型尚未训练，因此会生成无意义的文本。不过，你可以将它直接替换第 5 至第 7 章中的标准 GPT 模型，然后对其进行训练。

此外，这一实现使用了[另一个扩展章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768
```

```bash
uv run gpt_with_kv_sharing.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--n_kv_producing_layers 6
```

在这个小型 GPT 配置中，整个模型仍包含相同的前馈层和输出头。主要的内存差异在于有多少注意力层会将 K/V 张量存入缓存。
