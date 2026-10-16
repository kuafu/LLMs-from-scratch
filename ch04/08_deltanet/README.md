# 用于线性注意力的门控 DeltaNet

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 Transformer，它们采用了相对于上下文长度按线性而非二次方扩展的注意力机制替代方案。

如下图所示，Qwen3-Next 和 Kimi Linear 都采用 3:1 的比例：每三个采用线性门控 DeltaNet 变体的 Transformer 块，就配有一个使用完整注意力的块。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next 与 Kimi Linear 对比">



&nbsp;

## 简介与概述

门控 DeltaNet 是一种线性注意力变体，灵感来自循环神经网络，并采用了[《门控 Delta 网络：使用 Delta 规则改进 Mamba2》](https://arxiv.org/abs/2412.06464)论文中的门控机制。从某种意义上说，门控 DeltaNet 是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 本身是一种线性注意力机制。

Kimi Linear 使用 Kimi Delta Attention（KDA）机制改造了 Qwen3-Next 的线性注意力；KDA 本质上是对门控 DeltaNet 的改进。Qwen3-Next 使用标量门（每个注意力头一个值）来控制记忆衰减率，而 Kimi Linear 则将其替换为针对每个特征维度的逐通道门控。作者表示，这让模型能更好地控制记忆，进而改善长上下文推理。

此外，对于完整注意力层，Kimi Linear 将 Qwen3-Next 的门控注意力层（本质上是带输出门控的标准多头注意力层）替换为多头潜在注意力（MLA）。这与我们之前在 DeepSeek V3/R1 一节讨论的 MLA 机制相同，但额外加入了一个门。（回顾一下，MLA 会压缩键/值空间以缩小 KV 缓存。）

Kimi Linear 中的 MLA 没有使用门，这是有意为之，以便作者能更直接地将这种架构与标准 MLA 进行比较。不过，他们[表示](https://x.com/yzhang_cs/status/1984631714464088563)，计划在未来加入这一门。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，本扩展材料将重点介绍门控 DeltaNet 部分。


&nbsp;
## 门控注意力

在介绍门控 DeltaNet 本身之前，先简单说说门。如前图中 Qwen3-Next 架构的上半部分所示，Qwen3-Next 使用了“门控注意力”。这本质上是在常规完整注意力的基础上增加了一个 sigmoid 门。

下面的代码展示了一个简单的门控改动：我将它添加到第 3 章的 `MultiHeadAttention` 代码中，以作说明：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



可以看到，模型按常规方式计算注意力后，会从同一输入中生成一个独立的门控信号，对其应用 sigmoid 函数，使数值保持在 0 到 1 之间，再将其与注意力输出相乘。这样模型就能动态调节某些特征。Qwen3-Next 的开发者[表示](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)，这有助于提高训练稳定性：

> [...] 注意力输出门控机制有助于消除 Attention Sink 和 Massive Activation 等问题，从而确保模型整体的数值稳定性。


&nbsp;
## 门控 DeltaNet

那么，什么是门控 DeltaNet？门控 DeltaNet（*Gated Delta Network* 的简称）是 Qwen3-Next 的线性注意力层，旨在作为标准 softmax 注意力的替代方案。正如前面提到的，它采用自论文[《门控 Delta 网络：使用 Delta 规则改进 Mamba2》](https://arxiv.org/abs/2412.06464)。

门控 DeltaNet 最初是作为 Mamba2 的改进版本提出的，它将 Mamba2 的门控衰减机制与 Delta 规则结合起来。

Mamba 是一种状态空间模型（Transformer 的替代方案），这是一个值得以后单独介绍的大主题。

Delta 规则指的是计算新值与预测值之间的差值（delta，Δ），并用它更新作为记忆状态的隐藏状态（后文会进一步介绍）。

（旁注：读过经典机器学习文献的读者，可以把它看作受生物学启发的赫布学习：“一起放电的细胞会连接在一起。”它基本上是感知机更新规则和基于梯度下降的学习方法的前身，但没有监督。）

门控 DeltaNet 也有一个门，类似于前面讨论的门控注意力中的门，但如图所示，它使用 SiLU 而不是逻辑 sigmoid 激活函数。（与标准 sigmoid 相比，选择 SiLU 可能有助于改善梯度流动和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="门控 DeltaNet" width=500px>

不过，如上图所示，门控 DeltaNet 中的“门控”还指几个额外的门：

- `α`（衰减门）控制记忆随时间衰减或重置的速度，
- `β`（更新门）控制新输入对状态的修改程度。

如下代码展示了上图中门控 DeltaNet 的简化实现（不含卷积混合）。这段代码参考了 Qwen3 团队的[官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)。

（请注意，一些实现将衰减门称为 `gk`（第 k 步的门），其中 `exp(gk)` 对应论文中的 $\alpha_t$。为了明确表示这种关系，下面的代码片段将对数空间中的门 `alpha_log` 与取指数后的衰减值 `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（为简化代码并聚焦于循环结构，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合。）

由上面的内容可以看出，门控 DeltaNet 与标准注意力（或门控注意力）有许多不同之处。

在门控注意力中，模型会计算所有 token 之间的常规注意力（每个 token 都会关注其他所有 token）。得到注意力输出后，再由一个门（sigmoid）决定保留多少输出。关键在于，它仍然是常规的缩放点积注意力，其计算量随上下文长度按二次方增长。

回顾一下，缩放点积注意力的计算方式是 softmax(QKᵀ)V，其中 Q 和 K 是 *n*-by-*d* 矩阵，*n* 表示输入 token 的数量，*d* 表示嵌入维度。因此，QKᵀ 会得到一个 *n*-by-*n* 的注意力矩阵，再与维度为 *n*-by-*d* 的值矩阵 V 相乘：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="二次复杂度注意力" width=500px />

门控 DeltaNet 没有 *n*-by-*n* 的注意力矩阵。相反，模型会逐个处理 token，并保留一个随着每个新 token 到来而更新的运行记忆（状态）。下面的代码实现了这一过程，其中 `S` 是在每个时间步 *t* 循环更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

这些门控制着记忆的变化方式：

- α（`alpha`）控制要遗忘（衰减）多少旧记忆。

- β（`beta`）控制时间步 *t* 的当前 token 对记忆的更新程度。

（上面代码片段中没有显示的最终输出门类似于门控注意力中的门，用于控制保留多少输出。）

因此，从某种意义上说，门控 DeltaNet 中的状态更新与循环神经网络（RNN）的工作方式类似。它的优点是随着上下文长度线性扩展（通过 for 循环实现），而非按二次方扩展。

这种循环状态更新的缺点是，与常规注意力（或门控注意力）相比，它牺牲了完整成对注意力所带来的全局上下文建模能力。

门控 DeltaNet 在一定程度上仍能捕捉上下文，但必须经过记忆（*S*）瓶颈。该记忆的大小固定，因此效率更高；但与 RNN 类似，它会将过去的上下文压缩到单个隐藏状态中。

这就是 Qwen3-Next 和 Kimi Linear 架构没有将所有注意力层替换为 DeltaNet 层，而是采用前面提到的 3:1 比例的原因。

&nbsp;
## DeltaNet 的内存节省

上一节讨论了 DeltaNet 相较于完整注意力的优势：其计算复杂度随上下文长度线性增长，而非二次方增长。

除了线性计算复杂度之外，DeltaNet 的另一个主要优势是节省内存，因为 DeltaNet 模块不会增大 KV 缓存。（有关 KV 缓存的更多信息，请参阅 [../03_kv-cache](../03_kv-cache)。）相反，如前所述，它们会保留一个大小固定的循环状态，因此内存用量不会随上下文长度变化。

对于常规多头注意力（MHA）层，我们可以按以下方式计算 KV 缓存大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（乘以 2 是因为缓存中同时存储了键和值。）

对于上面实现的简化版 DeltaNet，公式如下：


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

请注意，`KV_cache_DeltaNet` 的内存大小与上下文长度（`n_tokens`）无关。另外，我们只存储记忆状态 S，而不是分别存储键和值，因此 `2 × bytes` 就变成了 `bytes`。不过，这里有一个二次项 `d_head × d_head`，它来自以下状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

不过通常无需担心这一点，因为头维度一般相对较小。例如，Qwen3-Next 中该维度为 128。

带有卷积混合的完整版本稍微复杂一些，还涉及卷积核大小等因素，不过上面的公式应当能说明门控 DeltaNet 背后的主要趋势和动机。

我们可以通过以下辅助脚本，将不同上下文长度下的内存估算和节省效果可视化：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

请注意，上述脚本按 `emb_dim / n_heads` 计算 `head_dim`。也就是说，2048 / 16  = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="门控 DeltaNet 的扩展趋势" width=500px>
