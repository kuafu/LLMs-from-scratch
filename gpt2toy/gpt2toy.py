"""GPT 核心算法：嵌入、Transformer 和贪婪自回归生成。

可选的 trace(event, **values) 同步观察实际计算得到的张量；默认不采集。
回调应只读观察，不修改张量或随机数状态。教学演示见 run_gpt2toy.py。

形状约定：B 为批大小（Batch Size），T 为序列长度（Sequence Length），
D 为嵌入维度（Embedding Dimension），H 为注意力头数，Vocab 为词表大小。
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """因果多头自注意力（Causal Multi-Head Self-Attention，MHA）。

    由同一输入产生 Q/K/V（Query/Key/Value，查询/键/值），每个头独立计算
    缩放点积注意力（Scaled Dot-Product Attention，SDPA），再拼接并投影。
    输入 [B,T,d_in]，输出 [B,T,d_out]；每头维度 d_h = d_out / H。
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """创建 Q/K/V 投影、输出投影，以及阻止读取未来位置的因果掩码。"""
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Buffer 随模型移动和保存；掩码不是可学习参数，上三角 j > i 表示未来位置。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, trace=None):
        """执行 Q/K/V 投影、分头、因果注意力与多头合并，可选 trace 观察实际张量。"""
        if trace is not None:
            trace("attention_start", x=x)
        b, num_tokens, _ = x.shape
        # 线性投影（Linear Projection）：Q 用于发起查询，K 用于匹配，V 提供聚合内容。
        keys_merged = self.W_key(x)
        queries_merged = self.W_query(x)
        values_merged = self.W_value(x)

        # 分头（Split Heads）：[B,T,d_out] -> [B,H,T,d_h]，各头保留完整序列。
        keys = keys_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA：Q @ K^T 产生 [B,H,T,T] 得分；因果掩码（Causal Mask）遮蔽未来。
        raw_scores = queries @ keys.transpose(2, 3)
        scale_factor = keys.shape[-1] ** 0.5
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = raw_scores.clone()
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        scaled_scores = attn_scores / scale_factor
        # 除以 sqrt(d_h) 控制点积尺度；Softmax 沿 Key 维归一化，随后应用注意力 Dropout。
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_per_head = attn_weights @ values
        # 加权聚合（Weighted Sum）后拼接各头（Concatenation）；contiguous 保证 view 可合并维度。
        context_vec = context_per_head.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        out = self.out_proj(context_vec)

        if trace is not None:
            trace(
                "attention", module=self, x=x,
                keys_merged=keys_merged, queries_merged=queries_merged, values_merged=values_merged,
                keys=keys, queries=queries, values=values, raw_scores=raw_scores,
                mask_bool=mask_bool, scaled_scores=scaled_scores, attn_weights=attn_weights,
                context_per_head=context_per_head, context_vec=context_vec, out=out,
            )
        return out


class LayerNorm(nn.Module):
    """层归一化（Layer Normalization，LN）：各 Token 独立沿最后的特征维归一化。

    计算 x_hat = (x - mean) / sqrt(var + eps)，再施加可学习的仿射变换
    gamma * x_hat + beta；这里 gamma/beta 对应 scale/shift，保持输入形状。
    """

    def __init__(self, emb_dim):
        """初始化逐特征缩放 gamma、平移 beta 与数值稳定常数 eps。"""
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, trace=None):
        """归一化 [B,T,D] 的 D 维；使用总体方差，不在 Batch 或时间维上混合统计量。"""
        if trace is not None:
            trace("layer_norm_start", x=x)
        # 每个 Token 独立地沿嵌入维度归一化，使用有偏方差。
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        norm_x = (x - mean) / std
        out = self.scale * norm_x + self.shift
        if trace is not None:
            trace("layer_norm", x=x, mean=mean, var=var, std=std, norm_x=norm_x, out=out, eps=self.eps)
        return out


class GELU(nn.Module):
    """高斯误差线性单元（Gaussian Error Linear Unit，GELU）的 tanh 近似。

    对每个元素施加非线性激活，不改变张量形状，也不包含可学习参数。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, trace=None):
        """用 tanh 近似标准正态累积分布函数 Phi，逐元素计算 x * Phi(x)。"""
        if trace is not None:
            trace("gelu_start", x=x)
        out = 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        if trace is not None:
            trace("gelu", out=out)
        return out


class FeedForward(nn.Module):
    """逐位置前馈网络（Position-wise Feed-Forward Network，FFN）。

    用两层线性变换构成多层感知机（Multi-Layer Perceptron，MLP）：D -> 4D -> D，
    中间使用 GELU。所有 Token 共享参数，但在此子层中彼此不交换信息。
    """

    def __init__(self, cfg):
        """建立升维线性层、GELU 激活与降维线性层。"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x, trace=None):
        """独立变换每个 Token 的特征，保留 [B,T,D] 的输入输出形状。"""
        if trace is not None:
            trace("feed_forward_start", x=x)
            trace("feed_forward_linear_start", layer_idx=0, x=x)
        h1 = self.layers[0](x)
        if trace is not None:
            trace("feed_forward_linear_end", layer_idx=0, out=h1)
        h2 = self.layers[1](h1, trace=trace)
        if trace is not None:
            trace("feed_forward_linear_start", layer_idx=2, x=h2)
        out = self.layers[2](h2)
        if trace is not None:
            trace("feed_forward_linear_end", layer_idx=2, out=out)
            trace("feed_forward", x=x, h1=h1, h2=h2, out=out)
        return out


class TransformerBlock(nn.Module):
    """采用前置层归一化（Pre-Layer Normalization，Pre-LN）的解码器块。

    因果 MHA 聚合可见上下文，FFN 变换逐位置特征；每个子层都使用
    残差连接（Residual Connection）把恒等旁路（Identity Shortcut）与分支输出相加。
    """

    def __init__(self, cfg):
        """组装 MHA、FFN、两层 LN，以及残差分支上的 Dropout。"""
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, trace=None):
        """执行两次 Pre-LN 子层变换与残差相加，始终保持残差流形状 [B,T,D]。"""
        if trace is not None:
            trace("block_start", x=x)

        # 第一条残差更新：x1 = x + Dropout(MHA(LN1(x)))，旁路保留归一化前的 x。
        shortcut1 = x
        x_norm1 = self.norm1(x, trace=trace)
        att_out = self.drop_shortcut(self.att(x_norm1, trace=trace))
        x_after_res1 = shortcut1 + att_out
        if trace is not None:
            trace("residual", name="res1", shortcut=shortcut1, sublayer_out=att_out, out=x_after_res1)

        # 第二条残差更新：x2 = x1 + Dropout(FFN(LN2(x1)))。
        shortcut2 = x_after_res1
        x_norm2 = self.norm2(x_after_res1, trace=trace)
        ff_out = self.drop_shortcut(self.ff(x_norm2, trace=trace))
        x_out = shortcut2 + ff_out
        if trace is not None:
            trace("residual", name="res2", shortcut=shortcut2, sublayer_out=ff_out, out=x_out)
            trace("block_end", out=x_out)
        return x_out


class GPTModel(nn.Module):
    """GPT（Generative Pre-trained Transformer，生成式预训练 Transformer）结构。

    采用仅解码器（Decoder-only）架构：词元嵌入与可学习绝对位置嵌入相加，
    经多个解码器块、最终 LN 和语言建模头（Language Modeling Head，LM Head）
    得到各位置的词表得分。此 Toy 模型随机初始化，不加载预训练权重。
    """

    def __init__(self, cfg):
        """按配置建立嵌入、解码器块和 LM Head；词元嵌入与输出头使用独立权重。"""
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, trace=None):
        """将整数 Token IDs [B,T] 映射为未归一化的词表得分 logits [B,T,Vocab]。

        T 不超过位置表容量。此处输出全部位置；生成循环随后选取末尾位置。
        可选 trace 接收入口、各计算阶段与出口事件，用于教学打印和快照。
        """
        _, seq_len = in_idx.shape
        if trace is not None:
            # begin：教学 trace 在此打印 GPTModel.forward 的入口。
            trace("model_start", in_idx=in_idx, training=self.training)

        tok_embeds = self.tok_emb(in_idx)
        # Learned Absolute Positional Embedding（可学习绝对位置嵌入）使用局部位置 0..T-1。
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)
        # 广播相加（Broadcast Addition）：[B,T,D] + [T,D] -> [B,T,D]。
        x_emb = tok_embeds + pos_embeds
        x = self.drop_emb(x_emb)
        if trace is not None:
            trace(
                "embeddings", module=self, in_idx=in_idx, pos_indices=pos_indices,
                tok_embeds=tok_embeds, pos_embeds=pos_embeds, x_emb=x_emb, x=x,
            )
            trace("blocks_start", num_blocks=len(self.trf_blocks))

        # 逐层传递隐藏状态（Hidden States）；各解码器块都有独立的可学习参数。
        for block in self.trf_blocks:
            x = block(x, trace=trace)

        if trace is not None:
            trace("blocks_end", num_blocks=len(self.trf_blocks))
        x_norm_final = self.final_norm(x, trace=trace)
        # LM Head：Linear(D, Vocab) 把隐藏特征投影为 logits，此处不施加 Softmax。
        logits = self.out_head(x_norm_final)
        if trace is not None:
            # end：教学 trace 在此打印出口，随后返回 logits。
            trace("model_end", module=self, in_idx=in_idx, logits=logits)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size, trace=None):
    """自回归生成（Autoregressive Generation）与贪婪解码（Greedy Decoding）。

    每次裁剪上下文、预测下一个 Token，再把选中的 ID 追加到 [B,T] 序列。
    每步重算窗口，不使用键值缓存（Key-Value Cache，KV Cache）。生成固定的
    max_new_tokens 步；遇到 <|endoftext|> 不会自动提前停止。
    """
    for step in range(max_new_tokens):
        # 滑动窗口（Sliding Window）：窗口移位后，模型会从 0 重新编号局部位置。
        idx_cond = idx[:, -context_size:]
        if trace is not None:
            trace("generation_start", step=step, max_new_tokens=max_new_tokens, idx=idx, idx_cond=idx_cond)
        # no_grad 关闭梯度记录；Dropout 是否启用仍取决于调用者设置的 train/eval 模式。
        with torch.no_grad():
            logits = model(idx_cond, trace=trace) if trace is not None else model(idx_cond)
        logits_last = logits[:, -1, :]
        # 下一个 Token 预测（Next-Token Prediction，NTP）：贪婪选择无需先计算 Softmax。
        idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        if trace is not None:
            trace("generation_end", step=step, logits_last=logits_last, idx_next=idx_next, idx=idx)
    return idx
