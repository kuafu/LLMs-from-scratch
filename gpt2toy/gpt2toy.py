"""GPT 核心算法：嵌入、Transformer 和贪婪自回归生成。

可选的 trace(event, **values) 同步观察实际计算得到的张量；默认不采集。
回调应只读观察，不修改张量或随机数状态。教学演示见 run_gpt2toy.py。
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
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
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, trace=None):
        b, num_tokens, _ = x.shape
        keys_merged = self.W_key(x)
        queries_merged = self.W_query(x)
        values_merged = self.W_value(x)

        # (B, T, D) -> (B, Heads, T, HeadDim)
        keys = keys_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values_merged.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 因果注意力：Q @ K^T，遮蔽未来位置，再除以 sqrt(HeadDim)。
        raw_scores = queries @ keys.transpose(2, 3)
        scale_factor = keys.shape[-1] ** 0.5
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = raw_scores.clone()
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        scaled_scores = attn_scores / scale_factor
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_per_head = attn_weights @ values
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
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, trace=None):
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x, trace=None):
        h1 = self.layers[0](x)
        h2 = self.layers[1](h1)
        out = self.layers[2](h2)
        if trace is not None:
            trace("feed_forward", x=x, h1=h1, h2=h2, out=out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
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
        if trace is not None:
            trace("block_start", x=x)

        # Pre-LN -> 多头注意力 -> Dropout -> 残差连接。
        shortcut1 = x
        x_norm1 = self.norm1(x, trace=trace)
        att_out = self.drop_shortcut(self.att(x_norm1, trace=trace))
        x_after_res1 = shortcut1 + att_out
        if trace is not None:
            trace("residual", name="res1", shortcut=shortcut1, sublayer_out=att_out, out=x_after_res1)

        # Pre-LN -> 逐位置前馈网络 -> Dropout -> 残差连接。
        shortcut2 = x_after_res1
        x_norm2 = self.norm2(x_after_res1, trace=trace)
        ff_out = self.drop_shortcut(self.ff(x_norm2, trace=trace))
        x_out = shortcut2 + ff_out
        if trace is not None:
            trace("residual", name="res2", shortcut=shortcut2, sublayer_out=ff_out, out=x_out)
            trace("block_end")
        return x_out


class GPTModel(nn.Module):
    def __init__(self, cfg):
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
        _, seq_len = in_idx.shape
        if trace is not None:
            trace("model_start", in_idx=in_idx, training=self.training)

        tok_embeds = self.tok_emb(in_idx)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)
        x_emb = tok_embeds + pos_embeds
        x = self.drop_emb(x_emb)
        if trace is not None:
            trace(
                "embeddings", module=self, in_idx=in_idx, pos_indices=pos_indices,
                tok_embeds=tok_embeds, pos_embeds=pos_embeds, x_emb=x_emb, x=x,
            )
            trace("blocks_start", num_blocks=len(self.trf_blocks))

        for block in self.trf_blocks:
            x = block(x, trace=trace)

        if trace is not None:
            trace("blocks_end", num_blocks=len(self.trf_blocks))
        x_norm_final = self.final_norm(x, trace=trace)
        logits = self.out_head(x_norm_final)
        if trace is not None:
            trace("model_end", module=self, in_idx=in_idx, logits=logits)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size, trace=None):
    """裁剪上下文，预测末尾位置并贪婪追加 Token ID；支持整个 batch。"""
    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        if trace is not None:
            trace("generation_start", step=step, max_new_tokens=max_new_tokens, idx=idx, idx_cond=idx_cond)
        with torch.no_grad():
            logits = model(idx_cond, trace=trace) if trace is not None else model(idx_cond)
        logits_last = logits[:, -1, :]
        idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        if trace is not None:
            trace("generation_end", step=step, logits_last=logits_last, idx_next=idx_next, idx=idx)
    return idx
