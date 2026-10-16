# 专家混合（Mixture of Experts，MoE）

本扩展材料展示了使用专家混合（MoE）层替代常规前馈（FFN）层时，每个 token 可以节省多少内存。



&nbsp;
## 简介

MoE 的核心思路是将 Transformer 块中的每个前馈模块替换为多个专家层，而每个专家层本身也是一个前馈模块。也就是说，我们用多个前馈块替换单个前馈块，如下图所示。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="滑动窗口注意力（SWA）" width="800px" />

Transformer 块中的前馈块（在上图中以深灰色方块表示）通常包含模型总参数量中的很大一部分。（请注意，Transformer 块以及其中的前馈块会在 LLM 中重复多次；DeepSeek-V3 中重复了 61 次。）

因此，将*单个*前馈块替换为*多个*前馈块（即采用 MoE 结构）会大幅增加模型的总参数量。不过，关键在于我们不会为每个 token 使用（“激活”）所有专家，而是由路由器为每个 token 只选择一小部分专家。

由于每次只有少数专家处于激活状态，MoE 模块通常被称为*稀疏*模块；与之相对，*稠密*模块总是使用完整的参数集。不过，MoE 带来的大量总参数可以提高 LLM 的容量，使其能在训练期间吸收更多知识。同时，稀疏性也能保持推理效率，因为所有参数不会同时使用。

例如，DeepSeek-V3 的每个 MoE 模块有 256 个专家，模型总参数量为 671 billion。但在推理期间，每次只有 9 个专家处于激活状态（1 个共享专家，加上路由器选出的 8 个专家）。因此，每个 token 的推理步骤只会使用 37 billion 个参数，而不是全部 671 billion。

DeepSeek-V3 的 MoE 设计有一个显著特点：使用共享专家，也就是对每个 token 始终处于激活状态的专家。这一思路并不新，早在 [2022 年的 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 年的 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就已提出。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE 共享专家" width="500px" />

（图示经过标注，源自论文 [DeepSeekMoE：迈向专家混合语言模型中的极致专家专门化](https://arxiv.org/abs/2401.06066)。）

&nbsp;

共享专家的优点最早在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中得到关注。研究发现，与不使用共享专家相比，使用共享专家可以提高整体建模性能。这可能是因为多个独立专家无需重复学习常见或重复模式，从而有更多空间学习更专门的模式。

&nbsp;
## 专家混合（MoE）的内存节省

MoE 模型节省内存的主要原因是激活值存储和计算量减少。在常规（稠密）前馈层（FFN）中，每个 token 都会激活完整的中间维度。

相比之下，MoE 层会将每个 token 路由到少数几个专家（例如，从 `num_experts` 个专家中选出 `top_k` 个）。

使用 MoE 层时，每个 token 只会激活 `top_k` 个专家。因此，相较于总容量相同的稠密 FFN，有效内存用量（和计算量）大约会按 `top_k / num_experts` 的比例缩小。


你可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本，为不同的模型配置应用该公式，查看使用 MoE 相较于 FFN 能节省多少内存（请注意，这里计算的是单个 Transformer 块；要得到总节省量，需要乘以模型中的 Transformer 块数）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

根据上面的结果可以看出，如果 FFN 的输入/输出维度（`emb_dim`）为 7,168，中间层大小（`hidden_dim`）为 14,336，那么该层大约有 ~308M 个参数，并且这些参数在前向传播时都会被激活。

现在，如果我们使用一个总参数量大致相同（~308M）的 MoE 层，其中有 8 个专家、每次激活 2 个专家，那么每次前向传播中只有 ~77M 个参数处于激活状态。

此外，在其他条件相同的情况下，专家数量越多，处于激活状态的参数就越少，“节省”也就越多：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="滑动窗口注意力（SWA）" width="500px" />



&nbsp;

你可以通过以下方式复现此图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了实践示例，用于在 GPT 模型实现中比较常规 FFN 和 MoE 的内存用量。请注意，如本页第一幅图所示，这两个脚本都使用了 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块（GPT-2 传统上使用 GELU）。

**注意：模型尚未训练，因此会生成无意义的文本。你可以在扩展材料 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 中找到一个训练好的 MoE。**



首先，运行使用常规 FFN 的模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平比较，我们需要缩小专家的大小。例如，如果使用 32 个专家，就需要将参数设为 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

可以看到，稠密前馈层处理一个 token 大约需要 0.76 ms，并使用约 0.19 MB 的激活值（峰值接近 0.75 MB）；

稀疏 MoE 层只占用约 0.04 MB 内存（峰值为 0.11 MB）。不过，代价是计算时间大约翻倍。（路由本身会带来额外开销，而且我的实现可能也不是最高效的。）

两种情况下，整体生成过程的 GPU 内存峰值仍约为 11.5 GB，因为两个版本加载了相同数量的权重参数，并且 KV 缓存大小相同；这两项是这里的主要开销。

无论如何，我们都能看到其中的权衡：MoE 将 FFN 内存用量减少了约 4-5×，同时使前馈计算时间大约翻倍。

请注意，如果一次处理更多 token，例如使用大于 1 的批量大小（这里为简化代码没有使用批量），节省效果会更加明显。



