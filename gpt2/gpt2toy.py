# 本文件汇总了我们在第2至4章 [Chapters 2-4] 中介绍的所有相关代码。
# 本文件可作为独立脚本 [Standalone Script] 直接运行。
# 本次调整：
#   1. 完整打印模型所有模块的权重矩阵与参数数据 (tok_emb, pos_emb, TransformerBlock 内各层权重、final_norm, out_head 等)
#   2. 树状展示前向传播中各模块的组织关系、张量维度变化与中间关键数值（查表结果、注意力权重矩阵等）

import re
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 确保在 Windows 控制台或各种终端环境下支持 UTF-8 打印
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


#####################################
# 格式化打印辅助工具 [Debug Formatting Helpers]
#####################################

def _fmt_num(val, precision=4):
    """格式化单个浮点数，兼容 -inf / inf"""
    if math.isinf(val):
        return f"{'-inf' if val < 0 else 'inf':>{precision + 3}}"
    return f"{val:>{precision + 3}.{precision}f}"


def format_1d(vec, precision=4):
    """将 1D 张量或列表格式化为对齐的单行字符串"""
    vals = vec.detach().cpu().tolist() if isinstance(vec, torch.Tensor) else list(vec)
    return "[" + ", ".join(_fmt_num(v, precision) for v in vals) + "]"


def print_1d_tensor(vec, indent="", label="", chunk_size=8, precision=4):
    """打印 1D 向量（若元素较多则按 chunk_size 自动分行对齐）"""
    vals = vec.detach().cpu().tolist() if isinstance(vec, torch.Tensor) else list(vec)
    n = len(vals)
    prefix = f"{label}: " if label else ""
    if n <= chunk_size:
        print(f"{indent}{prefix}[{', '.join(_fmt_num(v, precision) for v in vals)}]")
    else:
        if label:
            print(f"{indent}{label} (共 {n} 维):")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_str = ", ".join(_fmt_num(v, precision) for v in vals[start:end])
            print(f"{indent}  [{start:02d}..{end - 1:02d}]: [{chunk_str}]")


def print_2d_tensor(mat, indent="", row_labels=None, chunk_cols=8, precision=4):
    """打印 2D 矩阵，支持行标签注解及宽矩阵自动分块换行"""
    rows = mat.detach().cpu().tolist() if isinstance(mat, torch.Tensor) else mat
    num_rows = len(rows)
    num_cols = len(rows[0]) if num_rows > 0 else 0

    if row_labels is None:
        row_labels = [f"Row {r:02d}" for r in range(num_rows)]
    max_label_len = max((len(lbl) for lbl in row_labels), default=0)

    for r_idx, row in enumerate(rows):
        lbl = row_labels[r_idx].ljust(max_label_len)
        if num_cols <= chunk_cols:
            row_str = ", ".join(_fmt_num(v, precision) for v in row)
            print(f"{indent}{lbl}: [{row_str}]")
        else:
            for c_start in range(0, num_cols, chunk_cols):
                c_end = min(c_start + chunk_cols, num_cols)
                chunk_str = ", ".join(_fmt_num(v, precision) for v in row[c_start:c_end])
                head_lbl = lbl if c_start == 0 else (" " * max_label_len)
                print(f"{indent}{head_lbl} [{c_start:02d}..{c_end - 1:02d}]: [{chunk_str}]")


#####################################
# 第2章 [Chapter 2]
#####################################

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str.get(i, "<|unk|>") for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词 [Tokenize]
        token_ids = tokenizer.encode(txt)

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


def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
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
        self.head_dim = d_out // num_heads  # 单个注意力头维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 合并各头输出的投影层 (默认含 bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, verbose=False, show_data=False, token_labels=None):
        b, num_tokens, d_in = x.shape
        t_labels = token_labels or [f"t{i}" for i in range(num_tokens)]

        if verbose:
            print(f"  │   │   ├── [MHA 子层: 因果多头注意力机制 (Multi-Head Attention)]")
            print(f"  │   │   │   ├── 1. Q/K/V 线性映射: (B={b}, T={num_tokens}, d_in={d_in}) -> 3 × (B={b}, T={num_tokens}, d_out={self.d_out}) [数值见下方分头展示]")

        keys = self.W_key(x)      # (b, num_tokens, d_out)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        values = self.W_value(x)   # (b, num_tokens, d_out)

        # 拆分为多头并转置: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if verbose:
            print(f"  │   │   │   ├── 2. 多头空间拆分: 划分为 {self.num_heads} 个头, 每头维度 head_dim={self.head_dim}")
            print(f"  │   │   │   │      Q, K, V 形状变为: (B={b}, Heads={self.num_heads}, T={num_tokens}, HeadDim={self.head_dim})")
            if show_data:
                max_t_len = max((len(lbl) for lbl in t_labels), default=0)
                for h in range(self.num_heads):
                    print(f"  │   │   │   │      [Head {h}] Queries / Keys / Values (形状 {num_tokens}×{self.head_dim}):")
                    for t_idx in range(num_tokens):
                        q_str = format_1d(queries[0, h, t_idx])
                        k_str = format_1d(keys[0, h, t_idx])
                        v_str = format_1d(values[0, h, t_idx])
                        lbl = t_labels[t_idx].ljust(max_t_len)
                        print(f"  │   │   │   │        {lbl} | Q={q_str}  K={k_str}  V={v_str}")

        # 计算点积注意力得分: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 因果掩码裁剪与填充 (上三角遮蔽为 -inf)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax 归一化为注意力概率权重
        scale_factor = keys.shape[-1] ** 0.5
        scaled_scores = attn_scores / scale_factor
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if verbose:
            print(f"  │   │   │   ├── 3. 缩放点积得分 (Q @ K^T / sqrt({self.head_dim})) 与因果掩码 -> Softmax 注意力权重:")
            print(f"  │   │   │   │      矩阵形状: (B={b}, Heads={self.num_heads}, T={num_tokens}, T={num_tokens})")
            if show_data:
                for h in range(self.num_heads):
                    print(f"  │   │   │   │      [Head {h}] 缩放及因果掩码后得分 (scaled_scores, 上三角 -inf 遮蔽未来词元):")
                    print_2d_tensor(scaled_scores[0, h], indent="  │   │   │   │        ", row_labels=t_labels)
                    print(f"  │   │   │   │      [Head {h}] Softmax 注意力权重矩阵 (attn_weights, 每行概率和=1.0):")
                    print_2d_tensor(attn_weights[0, h], indent="  │   │   │   │        ", row_labels=t_labels)

        # 与 Values 加权求和并转置回 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并所有头: (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        if verbose:
            print(f"  │   │   │   ├── 4. 加权聚合与多头合拢: (attn_weights @ V) -> 恢复形状 (B={b}, T={num_tokens}, d_out={self.d_out})")
            if show_data:
                print_2d_tensor(context_vec[0], indent="  │   │   │   │        ", row_labels=t_labels)

        # 最终投影输出
        out = self.out_proj(context_vec)
        if verbose:
            print(f"  │   │   │   └── 5. 输出线性投影 (out_proj): 最终输出形状 (B={b}, T={num_tokens}, d_out={self.d_out})")
            if show_data:
                print_2d_tensor(out[0], indent="  │   │   │            ", row_labels=t_labels)

        return out


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

    def forward(self, x, verbose=False, show_data=False, token_labels=None):
        t_labels = token_labels or [f"t{i}" for i in range(x.shape[1])]
        if verbose:
            print(f"  │   │   ├── [FFN 子层: 逐位置前馈网络 (Feed-Forward Network)]")

        h1 = self.layers[0](x)
        h2 = self.layers[1](h1)
        out = self.layers[2](h2)

        if verbose:
            print(f"  │   │   │   ├── 1. 升维投影 Linear({x.shape[-1]} -> {h1.shape[-1]}): {tuple(x.shape)} -> {tuple(h1.shape)}")
            if show_data:
                print_2d_tensor(h1[0], indent="  │   │   │   │        ", row_labels=t_labels)
            print(f"  │   │   │   ├── 2. 非线性激活 GELU: 形状保持 {tuple(h2.shape)}")
            if show_data:
                print_2d_tensor(h2[0], indent="  │   │   │   │        ", row_labels=t_labels)
            print(f"  │   │   │   └── 3. 降维投影 Linear({h2.shape[-1]} -> {out.shape[-1]}): {tuple(h2.shape)} -> {tuple(out.shape)}")
            if show_data:
                print_2d_tensor(out[0], indent="  │   │   │            ", row_labels=t_labels)
        return out


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

    def forward(self, x, block_idx=0, verbose=False, show_data=False, token_labels=None):
        t_labels = token_labels or [f"t{i}" for i in range(x.shape[1])]
        if verbose:
            print(f"  │   ┌── [TransformerBlock {block_idx}] 输入张量: (B={x.shape[0]}, T={x.shape[1]}, emb_dim={x.shape[2]})")

        # 子层 1: Multi-Head Attention (Pre-LN + 残差连接)
        shortcut = x
        x_norm1 = self.norm1(x)
        if verbose:
            print(f"  │   │   ├── [Pre-LN 1] 归一化 LayerNorm(x): 形状保持 {tuple(x_norm1.shape)}")
            if show_data:
                print_2d_tensor(x_norm1[0], indent="  │   │   │     ", row_labels=t_labels)

        att_out = self.att(x_norm1, verbose=verbose, show_data=show_data, token_labels=t_labels)
        att_out = self.drop_shortcut(att_out)
        x = att_out + shortcut  # 残差连接 1: 将原始输入 shortcut 加回注意力输出
        if verbose:
            print(f"  │   │   ├── [残差连接 1 (Shortcut)] x = att_out + shortcut: 保持 {tuple(x.shape)}")
            if show_data:
                print_2d_tensor(x[0], indent="  │   │   │     ", row_labels=t_labels)

        # 子层 2: FeedForward Network (Pre-LN + 残差连接)
        shortcut = x
        x_norm2 = self.norm2(x)
        if verbose:
            print(f"  │   │   ├── [Pre-LN 2] 归一化 LayerNorm(x): 形状保持 {tuple(x_norm2.shape)}")
            if show_data:
                print_2d_tensor(x_norm2[0], indent="  │   │   │     ", row_labels=t_labels)

        ff_out = self.ff(x_norm2, verbose=verbose, show_data=show_data, token_labels=t_labels)
        ff_out = self.drop_shortcut(ff_out)
        x = ff_out + shortcut  # 残差连接 2: 将注意力阶段输出 shortcut 加回前馈输出
        if verbose:
            print(f"  │   │   └── [残差连接 2 (Shortcut)] x = ff_out + shortcut: 最终块输出 {tuple(x.shape)}")
            if show_data:
                print_2d_tensor(x[0], indent="  │   │         ", row_labels=t_labels)
            print(f"  │   └── [TransformerBlock {block_idx} 计算完成]\n  │")

        return x


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

    def forward(self, in_idx, verbose=False, show_data=False, token_labels=None):
        batch_size, seq_len = in_idx.shape
        t_labels = token_labels or [f"t{i}" for i in range(seq_len)]

        if verbose:
            print(f"  ┌── [1. 输入嵌入阶段 (Input Embeddings)]")
            print(f"  │   ├── 输入 Token IDs in_idx: {in_idx.tolist()} | 形状 (B={batch_size}, T={seq_len})")

        # 词嵌入与位置嵌入查表
        tok_embeds = self.tok_emb(in_idx)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)

        if verbose:
            print(f"  │   ├── 词嵌入查表 tok_emb(in_idx): 形状 (B={batch_size}, T={seq_len}, emb_dim={self.cfg['emb_dim']})")
            if show_data:
                print_2d_tensor(tok_embeds[0], indent="  │   │     ", row_labels=t_labels)

            print(f"  │   ├── 位置嵌入查表 pos_emb(0..{seq_len - 1}): 形状 (T={seq_len}, emb_dim={self.cfg['emb_dim']})")
            if show_data:
                pos_labels = [f"Pos={i} ({t_labels[i]})" for i in range(seq_len)]
                print_2d_tensor(pos_embeds, indent="  │   │     ", row_labels=pos_labels)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        if verbose:
            print(f"  │   └── 嵌入融合 x = tok_emb + pos_emb: 形状 {tuple(x.shape)}")
            if show_data:
                print_2d_tensor(x[0], indent="  │         ", row_labels=t_labels)
            print(f"  └── [输入嵌入阶段完毕]\n  │")

        # 逐层穿过 Transformer Blocks
        if verbose:
            print(f"  ┌── [2. Transformer 核心主干网络 (共 {len(self.trf_blocks)} 个 Block 堆叠)]")

        for i, block in enumerate(self.trf_blocks):
            x = block(x, block_idx=i, verbose=verbose, show_data=show_data, token_labels=t_labels)

        if verbose:
            print(f"  └── [Transformer 主干网络全部 {len(self.trf_blocks)} 个 Blocks 执行完毕]\n  │")

        # 最终层归一化与线性投影头
        if verbose:
            print(f"  ┌── [3. 输出预测阶段 (Final LayerNorm & Output Head)]")

        x = self.final_norm(x)
        if verbose:
            print(f"  │   ├── 最终层归一化 final_norm(x): 形状 (B={batch_size}, T={seq_len}, emb_dim={self.cfg['emb_dim']})")
            if show_data:
                print_2d_tensor(x[0], indent="  │   │     ", row_labels=t_labels)

        logits = self.out_head(x)
        if verbose:
            print(f"  │   └── 词表空间投影 out_head(x): (B={batch_size}, T={seq_len}, emb_dim={self.cfg['emb_dim']}) -> (B={batch_size}, T={seq_len}, vocab_size={self.cfg['vocab_size']})")
            if show_data:
                print(f"  │         [全序列每个位置对 {self.cfg['vocab_size']} 个词表 Token 的预测 Logits]:")
                print_2d_tensor(logits[0], indent="  │         ", row_labels=t_labels)
            print(f"  └── [输出预测阶段完毕 -> 生成全序列 Logits 张量]\n")

        return logits


#####################################
# 模型架构、参数统计与全量权重矩阵打印
#####################################

def print_model_architecture_and_weights(model, cfg, tokenizer, print_weights=True):
    """打印模型超参规格、精确参数量统计及全部关键权重矩阵数据"""
    vocab_labels = [f"ID={idx} ('{word}')" for word, idx in sorted(tokenizer.str_to_int.items(), key=lambda x: x[1])]
    pos_labels = [f"Pos={i}" for i in range(cfg["context_length"])]
    d = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    head_dim = d // n_heads

    # 从真实 PyTorch 模型提取精确参数计数
    tok_emb_p = sum(p.numel() for p in model.tok_emb.parameters())
    pos_emb_p = sum(p.numel() for p in model.pos_emb.parameters())
    b0 = model.trf_blocks[0]
    ln1_p = sum(p.numel() for p in b0.norm1.parameters())
    qkv_p = sum(p.numel() for m in (b0.att.W_query, b0.att.W_key, b0.att.W_value) for p in m.parameters())
    out_proj_p = sum(p.numel() for p in b0.att.out_proj.parameters())
    mha_p = sum(p.numel() for p in b0.att.parameters())
    ln2_p = sum(p.numel() for p in b0.norm2.parameters())
    ffn_fc1_p = sum(p.numel() for p in b0.ff.layers[0].parameters())
    ffn_fc2_p = sum(p.numel() for p in b0.ff.layers[2].parameters())
    ffn_p = sum(p.numel() for p in b0.ff.parameters())
    block_p = sum(p.numel() for p in b0.parameters())
    all_blocks_p = sum(p.numel() for p in model.trf_blocks.parameters())
    final_ln_p = sum(p.numel() for p in model.final_norm.parameters())
    out_head_p = sum(p.numel() for p in model.out_head.parameters())
    total_p = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 78)
    print("             GPT-2 Toy 模型配置与各模块参数量构成分析")
    print("=" * 78)
    print(f"  [超参数配置规格 (Hyperparameters)]:")
    print(f"   * 词表大小 (vocab_size)      : {cfg['vocab_size']} (微型演示词表)")
    print(f"   * 上下文长度 (context_length) : {cfg['context_length']} Tokens")
    print(f"   * 嵌入维度 (emb_dim)         : {d}")
    print(f"   * 注意力头数 (n_heads)        : {n_heads} (单头维度 head_dim = {d} // {n_heads} = {head_dim})")
    print(f"   * 前馈隐藏维 (4 * emb_dim)    : {4 * d}")
    print(f"   * Transformer 层数 (n_layers): {cfg['n_layers']}")
    print(f"   * QKV 偏置 (qkv_bias)        : {cfg['qkv_bias']}")
    print(f"   * Dropout 率 (drop_rate)     : {cfg['drop_rate']}")
    print("-" * 78)

    vocab_items = [f"'{w}': {idx}" for w, idx in tokenizer.str_to_int.items()]
    print(f"  [微型词表映射 (Vocab Mapping)]:")
    print("   " + ", ".join(vocab_items))
    print("-" * 78)

    print(f"  [各模块参数量组织分布 (Parameter Breakdown - 实测统计)]:")
    print(f"   1. 输入嵌入阶段 (Input Embeddings):")
    print(f"      - 词嵌入查找表 (tok_emb.weight)         : {tok_emb_p:>4}  ({cfg['vocab_size']} × {d})")
    print(f"      - 位置嵌入查找表 (pos_emb.weight)       : {pos_emb_p:>4}  ({cfg['context_length']} × {d})")
    print(f"   2. Transformer 核心主干 ({cfg['n_layers']} 层堆叠):")
    print(f"      - 单个 TransformerBlock 内部构成:")
    print(f"        * Pre-LN 1 (norm1: scale + shift)    : {ln1_p:>4}  ({d} + {d})")
    print(f"        * 多头注意力 Q/K/V 权重 (qkv_bias={cfg['qkv_bias']}): {qkv_p:>4}  (3 × {d}×{d})")
    print(f"        * 多头注意力输出投影 (out_proj: W + b) : {out_proj_p:>4}  ({d}×{d} + {d})")
    print(f"        * 多头注意力小计 (MHA Total)          : {mha_p:>4}")
    print(f"        * Pre-LN 2 (norm2: scale + shift)    : {ln2_p:>4}  ({d} + {d})")
    print(f"        * 前馈第 1 层升维 (Linear {d}->{4*d} + b)  : {ffn_fc1_p:>4}  ({d}×{4*d} + {4*d})")
    print(f"        * 前馈第 2 层降维 (Linear {4*d}->{d} + b)  : {ffn_fc2_p:>4}  ({4*d}×{d} + {d})")
    print(f"        * 前馈网络小计 (FFN Total)            : {ffn_p:>4}")
    print(f"      - 单个 Block 参数小计                   : {block_p:>4}")
    print(f"      - 全部 {cfg['n_layers']} 个 Blocks 累计参数量           : {all_blocks_p:>4}")
    print(f"   3. 输出预测阶段 (Output Head):")
    print(f"      - 最终层归一化 (final_norm: scale+shift): {final_ln_p:>4}  ({d} + {d})")
    print(f"      - 词表线性输出头 (out_head.weight, 无偏置): {out_head_p:>4}  ({cfg['vocab_size']} × {d})")
    print("   " + "-" * 66)
    print(f"   [Summary] 模型全模块可学习总参数量         : {total_p:>4}  (float32 约 {total_p * 4} Bytes)")
    print("=" * 78)

    if not print_weights:
        return

    print("\n" + "=" * 78)
    print("        GPT-2 Toy 全模块关键权重数据清单 (Initialized Weights)")
    print("        (随机种子 torch.manual_seed(123), PyTorch Linear 权重形状为 [out, in])")
    print("=" * 78)

    # 1. Embedding 权重
    print("\n  ┌── [权重组 1: 输入嵌入层权重 (tok_emb & pos_emb)]")
    print(f"  │   ├── tok_emb.weight | 形状: {tuple(model.tok_emb.weight.shape)} (vocab_size={cfg['vocab_size']}, emb_dim={d}):")
    print_2d_tensor(model.tok_emb.weight, indent="  │   │     ", row_labels=vocab_labels)

    print(f"  │   └── pos_emb.weight | 形状: {tuple(model.pos_emb.weight.shape)} (context_length={cfg['context_length']}, emb_dim={d}):")
    print_2d_tensor(model.pos_emb.weight, indent="  │         ", row_labels=pos_labels)
    print("  └── [输入嵌入层权重打印完毕]")

    # 2. Transformer Blocks 权重
    head_row_labels = [
        f"OutDim {r} (Head {r // head_dim}, d{r % head_dim})"
        for r in range(d)
    ]
    emb_row_labels = [f"OutDim {r}" for r in range(d)]

    for i, block in enumerate(model.trf_blocks):
        print(f"\n  ┌── [权重组 2.{i}: TransformerBlock {i} 内部权重]")

        # norm1
        print(f"  │   ├── [norm1 (Pre-LayerNorm 1)]")
        print_1d_tensor(block.norm1.scale, indent="  │   │     ", label=f"norm1.scale {tuple(block.norm1.scale.shape)}")
        print_1d_tensor(block.norm1.shift, indent="  │   │     ", label=f"norm1.shift {tuple(block.norm1.shift.shape)}")

        # MultiHeadAttention
        print(f"  │   ├── [att (MultiHeadAttention) | 计算公式: x @ W^T (+ b)]")
        print(f"  │   │   ├── att.W_query.weight | 形状: {tuple(block.att.W_query.weight.shape)} (bias={block.att.W_query.bias is not None}):")
        print_2d_tensor(block.att.W_query.weight, indent="  │   │   │     ", row_labels=head_row_labels)
        if block.att.W_query.bias is not None:
            print_1d_tensor(block.att.W_query.bias, indent="  │   │   │     ", label="att.W_query.bias")

        print(f"  │   │   ├── att.W_key.weight   | 形状: {tuple(block.att.W_key.weight.shape)} (bias={block.att.W_key.bias is not None}):")
        print_2d_tensor(block.att.W_key.weight, indent="  │   │   │     ", row_labels=head_row_labels)
        if block.att.W_key.bias is not None:
            print_1d_tensor(block.att.W_key.bias, indent="  │   │   │     ", label="att.W_key.bias")

        print(f"  │   │   ├── att.W_value.weight | 形状: {tuple(block.att.W_value.weight.shape)} (bias={block.att.W_value.bias is not None}):")
        print_2d_tensor(block.att.W_value.weight, indent="  │   │   │     ", row_labels=head_row_labels)
        if block.att.W_value.bias is not None:
            print_1d_tensor(block.att.W_value.bias, indent="  │   │   │     ", label="att.W_value.bias")

        print(f"  │   │   ├── att.out_proj.weight | 形状: {tuple(block.att.out_proj.weight.shape)}:")
        print_2d_tensor(block.att.out_proj.weight, indent="  │   │   │     ", row_labels=emb_row_labels)
        print_1d_tensor(block.att.out_proj.bias, indent="  │   │   │     ", label=f"att.out_proj.bias {tuple(block.att.out_proj.bias.shape)}")

        print(f"  │   │   └── att.mask (因果掩码 Buffer, 非训练参数) | 形状: {tuple(block.att.mask.shape)} (1 表示遮蔽为 -inf):")
        mask_int_rows = [[int(v) for v in row] for row in block.att.mask.tolist()]
        for r_idx, m_row in enumerate(mask_int_rows):
            print(f"  │   │         Query Pos={r_idx}: {m_row}")

        # norm2
        print(f"  │   ├── [norm2 (Pre-LayerNorm 2)]")
        print_1d_tensor(block.norm2.scale, indent="  │   │     ", label=f"norm2.scale {tuple(block.norm2.scale.shape)}")
        print_1d_tensor(block.norm2.shift, indent="  │   │     ", label=f"norm2.shift {tuple(block.norm2.shift.shape)}")

        # FeedForward
        fc1 = block.ff.layers[0]
        fc2 = block.ff.layers[2]
        print(f"  │   └── [ff (FeedForward: Linear({d}->{4*d}) -> GELU -> Linear({4*d}->{d}))]")
        print(f"  │       ├── ff.layers[0].weight (升维层) | 形状: {tuple(fc1.weight.shape)}:")
        fc1_labels = [f"HidDim {r:02d}" for r in range(fc1.weight.shape[0])]
        print_2d_tensor(fc1.weight, indent="  │       │     ", row_labels=fc1_labels)
        print_1d_tensor(fc1.bias, indent="  │       │     ", label=f"ff.layers[0].bias {tuple(fc1.bias.shape)}")

        print(f"  │       └── ff.layers[2].weight (降维层) | 形状: {tuple(fc2.weight.shape)}:")
        print_2d_tensor(fc2.weight, indent="  │             ", row_labels=emb_row_labels)
        print_1d_tensor(fc2.bias, indent="  │             ", label=f"ff.layers[2].bias {tuple(fc2.bias.shape)}")
        print(f"  └── [TransformerBlock {i} 权重打印完毕]")

    # 3. Final Norm & Output Head 权重
    print("\n  ┌── [权重组 3: 输出预测层权重 (final_norm & out_head)]")
    print(f"  │   ├── [final_norm (最终层归一化)]")
    print_1d_tensor(model.final_norm.scale, indent="  │   │     ", label=f"final_norm.scale {tuple(model.final_norm.scale.shape)}")
    print_1d_tensor(model.final_norm.shift, indent="  │   │     ", label=f"final_norm.shift {tuple(model.final_norm.shift.shape)}")

    print(f"  │   └── out_head.weight (词表投影矩阵, 无偏置) | 形状: {tuple(model.out_head.weight.shape)}:")
    print_2d_tensor(model.out_head.weight, indent="  │         ", row_labels=vocab_labels)
    print("  └── [输出预测层权重打印完毕]")
    print("=" * 78 + "\n")


#####################################
# 自回归生成循环与主入口
#####################################

def generate_text_simple(model, idx, max_new_tokens, context_size, tokenizer, verbose=True, show_data=True):
    """自回归文本生成循环，展示每次步进时的数据量、模块流动、中间激活值与预测决策"""
    for step in range(max_new_tokens):
        current_seq_len = idx.shape[1]
        current_ids = idx[0].tolist()
        current_text = tokenizer.decode(current_ids)

        print("\n" + "=" * 78)
        print(f"  [自回归生成步骤 (Step {step + 1}/{max_new_tokens})]  当前上下文序列长度 T = {current_seq_len}")
        print(f"  当前输入文本: \"{current_text}\" (Token IDs: {current_ids})")
        print("=" * 78)

        # 裁剪超出 context_size 的上下文
        idx_cond = idx[:, -context_size:]
        cond_ids = idx_cond[0].tolist()
        token_labels = [
            f"t{pos}[ID={tid}:'{tokenizer.int_to_str.get(tid, '<|unk|>')}']"
            for pos, tid in enumerate(cond_ids)
        ]

        # 前向传播 (启用详细模块组织、数据形状与关键张量数值打印)
        with torch.no_grad():
            logits = model(idx_cond, verbose=verbose, show_data=show_data, token_labels=token_labels)

        # 聚焦最后一个时间步
        logits_last = logits[:, -1, :]  # (batch=1, vocab_size)
        probs = torch.softmax(logits_last, dim=-1)[0]  # (vocab_size,)

        # 贪婪采样 (Argmax)
        idx_next = torch.argmax(logits_last, dim=-1, keepdim=True)  # (batch=1, 1)
        next_token_id = idx_next.item()
        next_word = tokenizer.decode([next_token_id])

        # 打印决策分析
        print(f"  ┌── [4. 自回归生成决策分析 (Next Token Prediction)]")
        print(f"  │   ├── 提取末尾时间步 Logits (logits[:, -1, :], 对应 {token_labels[-1]}): 形状 {tuple(logits_last.shape)}")
        print(f"  │   ├── 词表中各候选 Token 预测分布 (Softmax 概率):")
        for token_word, tid in sorted(tokenizer.str_to_int.items(), key=lambda x: x[1]):
            prob_percent = probs[tid].item() * 100
            logit_val = logits_last[0, tid].item()
            is_best = "  <-- [最高概率 / Argmax 选中]" if tid == next_token_id else ""
            word_disp = f"'{token_word}'".ljust(15)
            print(f"  │   │   * ID={tid} {word_disp}: {prob_percent:6.2f}%  [Logit: {logit_val:7.4f}]{is_best}")
        print(f"  │   ├── 贪婪采样结果: idx_next = {idx_next.tolist()} -> Token ID = {next_token_id} (\"{next_word}\")")

        # 将新 token 追加到序列中
        idx = torch.cat((idx, idx_next), dim=1)
        new_text = tokenizer.decode(idx[0].tolist())
        print(f"  │   └── 序列拼接更新: 长度 T 由 {current_seq_len} -> {idx.shape[1]} | 新序列 IDs: {idx[0].tolist()}")
        print(f"  └── [步骤 {step + 1} 完成] -> 当前累计生成文本: \"{new_text}\"")

    return idx


def main():
    # 自定义微型词表 (与 SimpleTokenizerV2 配合)
    toy_vocab = {
        "<|unk|>": 0,
        "<|endoftext|>": 1,
        "Hello": 2,
        ",": 3,
        "I": 4,
        "am": 5
    }
    tokenizer = SimpleTokenizerV2(toy_vocab)

    TOY_CONFIG = {
        "vocab_size": len(toy_vocab),  # 6
        "context_length": 6,           # 6
        "emb_dim": 4,                  # 4
        "n_heads": 2,                  # 2 (每头维度 head_dim = 2)
        "n_layers": 1,                 # 1
        "drop_rate": 0.0,              # 0.0 (推断模式)
        "qkv_bias": False
    }

    # 初始化固定随机种子的 Toy GPT 模型
    torch.manual_seed(123)
    model = GPTModel(TOY_CONFIG)
    model.eval()  # 禁用 Dropout

    # 1. 打印模型架构、参数构成统计以及 tok_emb / pos_emb 等全部模块关键权重数据
    print_model_architecture_and_weights(model, TOY_CONFIG, tokenizer, print_weights=True)

    # 2. 准备输入文本与编码张量
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print("=" * 78)
    print("                         [IN: 输入准备阶段]")
    print("=" * 78)
    print(f"  * 原始输入文本 (start_context) : \"{start_context}\"")
    print(f"  * Tokenizer 编码序列 (IDs)     : {encoded}")
    print(f"  * 逐词映射明细                 : " + ", ".join(f"'{tokenizer.int_to_str[i]}' -> {i}" for i in encoded))
    print(f"  * 封装为 Batch 张量            : {encoded_tensor.tolist()} | 形状: {tuple(encoded_tensor.shape)} (B=1, T={len(encoded)})")
    print("=" * 78)

    # 3. 自回归生成循环 (同时展示模块数据流与每步关键张量/注意力权重数值)
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=2,
        context_size=TOY_CONFIG["context_length"],
        tokenizer=tokenizer,
        verbose=True,
        show_data=True
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # 4. 最终生成输出总结
    print("\n" + "=" * 78)
    print("                         [OUT: 最终生成结果]")
    print("=" * 78)
    print(f"  * 最终输出 Token IDs 张量 : {out.tolist()}")
    print(f"  * 最终序列总长度          : {len(out[0])} Tokens")
    print(f"  * 最终反分词还原文本      : \"{decoded_text}\"")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
