# 本文件汇总了我们在第2至4章 [Chapters 2-4] 中介绍的所有相关代码。
# 本文件可作为独立脚本 [Standalone Script] 直接运行。

import sys

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 代码复制自 ../ch04/01_main-chapter-code/gpt.py

#####################################
# 第2章 [Chapter 2]
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词 [Tokenize]
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口 [Sliding Window] 将整本书切分为最大长度 [max_length] 的重叠序列 [Sequences]
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器 [Tokenizer]
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集 [Dataset]
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器 [DataLoader]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第3章 [Chapter 3]
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 缩减投影维度 [Projection Dimension] 以匹配期望的输出维度 [Output Dimension]

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 用于合并各头输出的线性层 [Linear Layer]
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # 形状 [Shape]: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过增加一个 `num_heads` 维度隐式拆分矩阵
        # 展开最后一个维度 [Dimension]: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置 [Transpose]: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 使用因果掩码 [Causal Mask] 计算缩放点积注意力 [Scaled Dot-Product Attention]（亦称自注意力 [Self-Attention]）
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积 [Dot Product]

        # 原始掩码 [Mask] 截断至词元数量 [Number of Tokens] 并转换为布尔型 [Boolean]
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码 [Mask] 填充注意力分数 [Attention Scores]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状 [Shape]: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并各头，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影 [Optional Projection]

        return context_vec


#####################################
# 第4章 [Chapter 4]
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


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

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块 [Attention Block] 的残差连接 / 捷径连接 [Shortcut Connection]
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状 [Shape]: [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回

        # 前馈网络块 [Feed-Forward Block] 的残差连接 / 捷径连接 [Shortcut Connection]
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList()
        for _ in range(cfg["n_layers"]):
            block = TransformerBlock(cfg)
            self.trf_blocks.append(block)

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # 形状 [Shape]: [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前上下文 [Context] 中索引 [Indices] 的 (B, T) 数组
    for _ in range(max_new_tokens):
        print(f"\n\n{50*'='}\n{22*' '}LOOP\n{50*'='}")

        # 如果超出支持的上下文大小 [Context Size]，则对当前上下文进行裁剪 [Crop]
        # 例如：如果大语言模型 [LLM] 仅支持 5 个词元 [Tokens]，而当前上下文大小为 10，
        # 那么仅最后 5 个词元会被用作上下文 [Context]
        idx_cond = idx[:, -context_size:]

        # 获取预测结果 [Predictions]
        with torch.no_grad():
            logits = model(idx_cond)

        # 仅关注最后一个时间步 [Time Step]
        # (batch, n_token, vocab_size) 变为 (batch, vocab_size)
        logits = logits[:, -1, :]
        print(f"logits.shape: {logits.shape}, logits: {logits}")
        print(f"logits[0]: {logits[0]}")

        # 获取词表 [Vocab] 中对数几率 / 未归一化概率 [Logits] 值最高的条目索引 [Index]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        print(f"idx_next.shape: {idx_next.shape}, idx_next: {idx_next}")
        print(f"idx_next[0]: {idx_next[0]}")

        # 将采样索引 [Sampled Index] 追加到正在生成的序列 [Sequence]
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        print(f"idx.shape: {idx.shape}, idx: {idx}")
        print(f"idx[0]: {idx[0]}")


    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词表大小 [Vocabulary Size]
        "context_length": 1024,  # 上下文长度 [Context Length]
        "emb_dim": 768,          # 嵌入维度 [Embedding Dimension]
        "n_heads": 12,           # 注意力头数量 [Number of Attention Heads]
        "n_layers": 12,          # 层数 [Number of Layers]
        "drop_rate": 0.1,        # 随机失活率 [Dropout Rate]
        "qkv_bias": False        # 查询-键-值偏置 [Query-Key-Value Bias]
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 禁用随机失活 [Disable Dropout]

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())


    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
