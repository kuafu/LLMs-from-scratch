"""GPT 教学诊断：终端打印、张量快照、参数统计和跨步剖析。

仅在显式传入 TeachingTrace 时采集；GPT 核心不依赖本模块。
"""

import math

import torch


class _TeeWriter:
    """将同一份输出同时写入终端和文本文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self._streams:
            stream.flush()


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


def print_residual_comparison(shortcut, sub_out, merged, indent="", row_labels=None, sub_name="sub_out"):
    """并排打印残差连接三列对照: shortcut + sub_out = merged"""
    s_rows = shortcut.detach().cpu().tolist()
    u_rows = sub_out.detach().cpu().tolist()
    m_rows = merged.detach().cpu().tolist()
    num_rows = len(s_rows)
    if row_labels is None:
        row_labels = [f"Row {r:02d}" for r in range(num_rows)]
    max_lbl = max((len(lbl) for lbl in row_labels), default=0)

    for r_idx in range(num_rows):
        lbl = row_labels[r_idx].ljust(max_lbl)
        s_str = format_1d(s_rows[r_idx])
        u_str = format_1d(u_rows[r_idx])
        m_str = format_1d(m_rows[r_idx])
        print(f"{indent}{lbl}: shortcut {s_str} + {sub_name} {u_str} = {m_str}")


def _inspect_attention(
    *, module, x, keys_merged, queries_merged, values_merged, keys, queries, values,
    raw_scores, mask_bool, scaled_scores, attn_weights, context_per_head, context_vec,
    out, verbose, show_data, token_labels, record_dict,
):
    b, num_tokens, d_in = x.shape
    t_labels = token_labels or [f"t{i}" for i in range(num_tokens)]
    max_t_len = max((len(lbl) for lbl in t_labels), default=0)
    scale_factor = keys.shape[-1] ** 0.5
    unmasked_scaled_scores = raw_scores / scale_factor

    if verbose:
        print(f"  │   │   ├── [MHA 子层: 因果多头注意力机制 (Multi-Head Attention)]")
        print(f"  │   │   │   ├── 1. Q/K/V 线性映射 (切头前完整 {module.d_out} 维): (B={b}, T={num_tokens}, d_in={d_in}) -> 3 × (B={b}, T={num_tokens}, d_out={module.d_out})")
        head_columns = ", ".join(
            f"列 [{h * module.head_dim}..{(h + 1) * module.head_dim - 1}] 对应 Head {h}"
            for h in range(module.num_heads)
        )
        print(f"  │   │   │   │      [注: {head_columns}]")
        if show_data:
            for t_idx in range(num_tokens):
                lbl = t_labels[t_idx].ljust(max_t_len)
                q_m = format_1d(queries_merged[0, t_idx])
                k_m = format_1d(keys_merged[0, t_idx])
                v_m = format_1d(values_merged[0, t_idx])
                print(f"  │   │   │   │        {lbl} | Q_full={q_m}  K_full={k_m}  V_full={v_m}")

    if verbose:
        print(f"  │   │   │   ├── 2. 多头空间拆分 (view + transpose): 划分为 {module.num_heads} 个独立注意力头, 每头维度 head_dim={module.head_dim}")
        print(f"  │   │   │   │      Q, K, V 形状变为: (B={b}, Heads={module.num_heads}, T={num_tokens}, HeadDim={module.head_dim})")
        if show_data:
            for h in range(module.num_heads):
                col_range = f"[{h * module.head_dim}..{(h + 1) * module.head_dim - 1}]"
                print(f"  │   │   │   │      [Head {h} (源自列 {col_range})] Queries / Keys / Values (形状 {num_tokens}×{module.head_dim}):")
                for t_idx in range(num_tokens):
                    q_str = format_1d(queries[0, h, t_idx])
                    k_str = format_1d(keys[0, h, t_idx])
                    v_str = format_1d(values[0, h, t_idx])
                    lbl = t_labels[t_idx].ljust(max_t_len)
                    print(f"  │   │   │   │        {lbl} | Q_{h}={q_str}  K_{h}={k_str}  V_{h}={v_str}")

    if verbose:
        print(f"  │   │   │   ├── 3. 注意力得分矩阵四步演化 (Raw Dot -> Scaled -> Causal Masked -> Softmax Weights):")
        print(f"  │   │   │   │      矩阵形状: (B={b}, Heads={module.num_heads}, T={num_tokens}, T={num_tokens}), 缩放因子 sqrt(d_k) = sqrt({module.head_dim}) ≈ {scale_factor:.4f}")
        if show_data:
            for h in range(module.num_heads):
                print(f"  │   │   │   │      [Head {h}] (3a) 原始点积得分 raw_scores = Q_{h} @ K_{h}^T:")
                print_2d_tensor(raw_scores[0, h], indent="  │   │   │   │        ", row_labels=t_labels)
                print(f"  │   │   │   │      [Head {h}] (3b) 缩放得分 unmasked_scaled = raw_scores / {scale_factor:.4f}:")
                print_2d_tensor(unmasked_scaled_scores[0, h], indent="  │   │   │   │        ", row_labels=t_labels)
                print(f"  │   │   │   │      [Head {h}] (3c) 因果掩码后得分 scaled_scores (上三角 j>i 置为 -inf 阻断未来信息):")
                print_2d_tensor(scaled_scores[0, h], indent="  │   │   │   │        ", row_labels=t_labels)
                print(f"  │   │   │   │      [Head {h}] (3d) Softmax / Dropout 后的实际注意力权重矩阵 attn_weights:")
                print_2d_tensor(attn_weights[0, h], indent="  │   │   │   │        ", row_labels=t_labels)
                print(f"  │   │   │   │        实际行和: {format_1d(attn_weights[0, h].sum(dim=-1))}")

    if verbose:
        print(f"  │   │   │   ├── 4. Value 加权聚合与多头拼接: context_h = attn_weights_h @ V_h -> 左右拼接恢复 (B={b}, T={num_tokens}, d_out={module.d_out})")
        if show_data:
            for t_idx in range(num_tokens):
                lbl = t_labels[t_idx].ljust(max_t_len)
                head_str = " || ".join(
                    f"Head{h}={format_1d(context_per_head[0, h, t_idx])}"
                    for h in range(module.num_heads)
                )
                cat_str = format_1d(context_vec[0, t_idx])
                print(f"  │   │   │   │        {lbl} | {head_str} => Concat={cat_str}")

    if verbose:
        print(f"  │   │   │   └── 5. 输出线性投影 (out_proj: context @ W_out^T + b_out): 最终输出形状 (B={b}, T={num_tokens}, d_out={module.d_out})")
        if show_data:
            print_2d_tensor(out[0], indent="  │   │   │            ", row_labels=t_labels)

    if record_dict is not None:
        heads_data = []
        for h in range(module.num_heads):
            heads_data.append({
                "head_idx": h,
                "q": queries[0, h].detach().cpu().tolist(),
                "k": keys[0, h].detach().cpu().tolist(),
                "v": values[0, h].detach().cpu().tolist(),
                "raw_scores": raw_scores[0, h].detach().cpu().tolist(),
                "unmasked_scaled_scores": unmasked_scaled_scores[0, h].detach().cpu().tolist(),
                "mask_bool": mask_bool.detach().cpu().tolist(),
                "masked_scores": scaled_scores[0, h].detach().cpu().tolist(),
                "attn_weights": attn_weights[0, h].detach().cpu().tolist(),
                "context_head": context_per_head[0, h].detach().cpu().tolist(),
            })
        record_dict.update({
            "scale_factor": float(scale_factor),
            "q_merged": queries_merged[0].detach().cpu().tolist(),
            "k_merged": keys_merged[0].detach().cpu().tolist(),
            "v_merged": values_merged[0].detach().cpu().tolist(),
            "heads": heads_data,
            "context_merged": context_vec[0].detach().cpu().tolist(),
            "out_proj_out": out[0].detach().cpu().tolist(),
        })


def _inspect_layer_norm(*, x, mean, var, std, norm_x, out, eps, verbose, show_data, token_labels, ln_name, indent, record_dict):
    post_mean = out.mean(dim=-1)
    post_var = out.var(dim=-1, unbiased=False)

    if verbose:
        t_labels = token_labels or [f"t{i}" for i in range(x.shape[1])]
        max_lbl = max((len(lbl) for lbl in t_labels), default=0)
        print(f"{indent}├── [{ln_name}] 逐位置归一化: (x - μ) / sqrt(σ² + {eps}) * scale + shift | 形状保持 {tuple(out.shape)}")
        if show_data:
            for t_idx in range(x.shape[1]):
                lbl = t_labels[t_idx].ljust(max_lbl)
                m_val = mean[0, t_idx, 0].item()
                v_val = var[0, t_idx, 0].item()
                s_val = std[0, t_idx, 0].item()
                out_str = format_1d(out[0, t_idx])
                pm_val = post_mean[0, t_idx].item()
                pv_val = post_var[0, t_idx].item()
                print(f"{indent}│     {lbl} | μ={m_val:7.4f}, σ²={v_val:7.4f}, std={s_val:7.4f} -> norm_out={out_str} (验证: μ'={pm_val:6.4f}, σ'²={pv_val:6.4f})")

    if record_dict is not None:
        record_dict.update({
            "eps": eps,
            "in": x[0].detach().cpu().tolist(),
            "mean": mean[0, :, 0].detach().cpu().tolist(),
            "var": var[0, :, 0].detach().cpu().tolist(),
            "std": std[0, :, 0].detach().cpu().tolist(),
            "norm_raw": norm_x[0].detach().cpu().tolist(),
            "out": out[0].detach().cpu().tolist(),
            "post_mean": post_mean[0].detach().cpu().tolist(),
            "post_var": post_var[0].detach().cpu().tolist(),
        })


def _inspect_feed_forward(*, x, h1, h2, out, verbose, show_data, token_labels, record_dict):
    t_labels = token_labels or [f"t{i}" for i in range(x.shape[1])]

    if verbose:
        print(f"  │   │   ├── [FFN 子层: 逐位置前馈网络 (Feed-Forward Network)]")

    if verbose:
        print(f"  │   │   │   ├── 1. 升维投影 Linear({x.shape[-1]} -> {h1.shape[-1]}): {tuple(x.shape)} -> {tuple(h1.shape)}")
        if show_data:
            print_2d_tensor(h1[0], indent="  │   │   │   │        ", row_labels=t_labels)
        print(f"  │   │   │   ├── 2. 非线性激活 GELU(h1): 形状保持 {tuple(h2.shape)}")
        if show_data:
            print_2d_tensor(h2[0], indent="  │   │   │   │        ", row_labels=t_labels)
        print(f"  │   │   │   └── 3. 降维投影 Linear({h2.shape[-1]} -> {out.shape[-1]}): {tuple(h2.shape)} -> {tuple(out.shape)}")
        if show_data:
            print_2d_tensor(out[0], indent="  │   │   │            ", row_labels=t_labels)

    if record_dict is not None:
        record_dict.update({
            "fc1_out": h1[0].detach().cpu().tolist(),
            "gelu_out": h2[0].detach().cpu().tolist(),
            "fc2_out": out[0].detach().cpu().tolist(),
        })


def _inspect_embeddings(*, module, in_idx, pos_indices, tok_embeds, pos_embeds, x_emb, x, verbose, show_data, token_labels, record_dict):
    batch_size, seq_len = in_idx.shape
    t_labels = token_labels
    max_lbl = max((len(lbl) for lbl in t_labels), default=0)

    if verbose:
        print(f"  ┌── [1. 输入嵌入阶段 (Input Embeddings)]")
        print(f"  │   ├── 输入 Token IDs in_idx: {in_idx.tolist()} | 形状 (B={batch_size}, T={seq_len})")

    if verbose:
        print(f"  │   ├── 词嵌入查表 tok_emb(in_idx): 从 tok_emb.weight ({module.tok_emb.num_embeddings}×{module.tok_emb.embedding_dim}) 按行抽取 -> 形状 (B={batch_size}, T={seq_len}, emb_dim={module.tok_emb.embedding_dim})")
        if show_data:
            print_2d_tensor(tok_embeds[0], indent="  │   │     ", row_labels=t_labels)

        print(f"  │   ├── 位置嵌入查表 pos_emb(0..{seq_len - 1}): 从 pos_emb.weight ({module.pos_emb.num_embeddings}×{module.pos_emb.embedding_dim}) 取前 {seq_len} 行 -> 形状 (T={seq_len}, emb_dim={module.pos_emb.embedding_dim})")
        if show_data:
            pos_labels = [f"Pos={i} ({t_labels[i]})" for i in range(seq_len)]
            print_2d_tensor(pos_embeds, indent="  │   │     ", row_labels=pos_labels)

    if verbose:
        print(f"  │   └── 嵌入广播相加融合 x = tok_emb + pos_emb: 形状 {tuple(x.shape)}")
        if show_data:
            for t_idx in range(seq_len):
                lbl = t_labels[t_idx].ljust(max_lbl)
                tid = in_idx[0, t_idx].item()
                t_str = format_1d(tok_embeds[0, t_idx])
                p_str = format_1d(pos_embeds[t_idx])
                x_str = format_1d(x[0, t_idx])
                print(f"  │         {lbl}: tok[ID={tid}] {t_str} + pos[Pos={t_idx}] {p_str} = {x_str}")
        print(f"  └── [输入嵌入阶段完毕]\n  │")

    record_dict.update({
        "in_idx": in_idx[0].detach().cpu().tolist(),
        "pos_indices": pos_indices.detach().cpu().tolist(),
        "tok_embeds": tok_embeds[0].detach().cpu().tolist(),
        "pos_embeds": pos_embeds.detach().cpu().tolist(),
        "x_emb": x_emb[0].detach().cpu().tolist(),
        "x_after_drop": x[0].detach().cpu().tolist(),
    })


def _inspect_output(*, module, in_idx, logits, verbose, show_data, token_labels, record_dict):
    batch_size, seq_len = in_idx.shape
    t_labels = token_labels
    probs_all = torch.softmax(logits, dim=-1)

    if verbose:
        print(f"  │   └── 词表空间投影 out_head(x): x_final_norm @ out_head.weight^T | (B={batch_size}, T={seq_len}, emb_dim={module.cfg['emb_dim']}) -> (B={batch_size}, T={seq_len}, vocab_size={module.cfg['vocab_size']})")
        if show_data:
            print(f"  │         [全序列每个位置对 {module.cfg['vocab_size']} 个词表 Token 的预测 Logits]:")
            print_2d_tensor(logits[0], indent="  │         ", row_labels=t_labels)
            print(f"  │         [全序列每个位置经 Softmax 归一化后的词表概率分布]:")
            print_2d_tensor(probs_all[0], indent="  │         ", row_labels=t_labels)
            print(f"  │         实际行和: {format_1d(probs_all[0].sum(dim=-1))}")
        print(f"  └── [输出预测阶段完毕 -> 生成全序列 Logits 张量]\n")

    record_dict.update({
        "logits_all": logits[0].detach().cpu().tolist(),
        "probs_all": probs_all[0].detach().cpu().tolist(),
    })


def _start_generation(tokenizer, step, max_new_tokens, idx, idx_cond, verbose):
    current_seq_len = idx.shape[1]
    current_ids = idx[0].tolist()
    current_text = tokenizer.decode(current_ids)

    if verbose:
        print("\n" + "=" * 78)
    if verbose:
        print(f"  [自回归生成步骤 (Step {step + 1}/{max_new_tokens})]  累计序列长度 = {current_seq_len}, 本步上下文 T = {idx_cond.shape[1]}")
    if verbose:
        print(f"  累计文本: \"{current_text}\" (Token IDs: {current_ids})")
        print(f"  本步实际输入: {tokenizer.decode(idx_cond[0].tolist())!r} (Token IDs: {idx_cond[0].tolist()})")
    if verbose:
        print("=" * 78)

    cond_ids = idx_cond[0].tolist()
    token_labels = [
        f"t{pos}[ID={tid}:'{tokenizer.int_to_str.get(tid, '<|unk|>')}']"
        for pos, tid in enumerate(cond_ids)
    ]

    step_rec = {
        "step_num": step + 1,
        "seq_len": len(cond_ids),
        "full_seq_len": current_seq_len,
        "context_start": current_seq_len - len(cond_ids),
        "input_ids": cond_ids,
        "input_tokens": [tokenizer.int_to_str.get(tid, "<|unk|>") for tid in cond_ids],
        "token_labels": token_labels,
        "input_text": tokenizer.decode(cond_ids),
        "full_text": current_text,
    }
    return step_rec


def _finish_generation(tokenizer, step, logits_last, idx_next, idx, step_rec, verbose):
    token_labels = step_rec["token_labels"]
    current_seq_len = idx.shape[1] - 1

    probs = torch.softmax(logits_last, dim=-1)[0]  # (vocab_size,)
    next_token_id = idx_next[0, 0].item()
    next_word = tokenizer.decode([next_token_id])

    # 打印决策分析
    if verbose:
        print(f"  ┌── [4. 自回归生成决策分析 (Next Token Prediction)]")
    if verbose:
        print(f"  │   ├── 提取末尾时间步 Logits (logits[:, -1, :], 对应 {token_labels[-1]}): 形状 {tuple(logits_last.shape)}")
    if verbose:
        print(f"  │   ├── 词表中各候选 Token 预测分布 (Softmax 概率):")
    candidates = []
    for token_word, tid in sorted(tokenizer.str_to_int.items(), key=lambda x: x[1]):
        prob_val = probs[tid].item()
        prob_percent = prob_val * 100
        logit_val = logits_last[0, tid].item()
        is_best = "  <-- [最高概率 / Argmax 选中]" if tid == next_token_id else ""
        word_disp = f"'{token_word}'".ljust(15)
        if verbose:
            print(f"  │   │   * ID={tid} {word_disp}: {prob_percent:6.2f}%  [Logit: {logit_val:7.4f}]{is_best}")
        candidates.append({
            "id": tid,
            "token": token_word,
            "logit": logit_val,
            "prob": prob_val,
            "is_selected": tid == next_token_id,
        })
    if verbose:
        print(f"  │   ├── 贪婪采样结果: idx_next = {idx_next.tolist()} -> Token ID = {next_token_id} (\"{next_word}\")")
    new_ids = idx[0].tolist()
    new_text = tokenizer.decode(new_ids)
    if verbose:
        print(f"  │   └── 序列拼接更新: 长度 T 由 {current_seq_len} -> {idx.shape[1]} | 新序列 IDs: {new_ids}")
    if verbose:
        print(f"  └── [步骤 {step + 1} 完成] -> 当前累计生成文本: \"{new_text}\"")

    step_rec["prediction"] = {
        "last_token_label": token_labels[-1],
        "logits_last": logits_last[0].detach().cpu().tolist(),
        "probs_last": probs.detach().cpu().tolist(),
        "candidates": candidates,
        "next_token_id": next_token_id,
        "next_token_str": next_word,
        "updated_ids": new_ids,
        "updated_text": new_text,
    }
    return step_rec


class TeachingTrace:
    """同步打印并保存首个 batch 样本的教学快照，不持有模型或计算图。

    ``record`` 是最近一次 forward 的独立 Python 数据；生成时可将每步数据
    追加到 ``steps_records``。同一个实例供一次顺序执行使用，不用于并发调用。
    """

    def __init__(self, tokenizer=None, *, verbose=True, show_data=True, steps_records=None):
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.show_data = show_data
        self.steps_records = steps_records
        self.record = {}
        self._labels = None
        self._block = None
        self._block_idx = -1
        self._norm_idx = 0
        self._generating = False

    @torch.no_grad()
    def __call__(self, event, **values):
        getattr(self, "_on_" + event)(**values)

    def _on_model_start(self, in_idx, training=False):
        if self.verbose:
            print(f"  >>> GPTModel.forward begin | in_idx.shape={tuple(in_idx.shape)}")
        if not self._generating:
            self.record = {}
        self.record["training"] = training
        self.record["embeddings"] = {}
        self.record["blocks"] = []
        self.record["output_stage"] = {}
        self._block = None
        self._block_idx = -1
        if self.tokenizer is None:
            self._labels = [f"t{i}" for i in range(in_idx.shape[1])]
        else:
            self._labels = [
                f"t{pos}[ID={tid}:'{self.tokenizer.int_to_str.get(tid, '<|unk|>')}']"
                for pos, tid in enumerate(in_idx[0].tolist())
            ]

    def _on_embeddings(self, **values):
        record = self.record["embeddings"] = {}
        _inspect_embeddings(**values, verbose=self.verbose, show_data=self.show_data,
                            token_labels=self._labels, record_dict=record)

    def _on_blocks_start(self, num_blocks):
        if self.verbose:
            print(f"  ┌── [2. Transformer 核心主干网络 (共 {num_blocks} 个 Block 堆叠)]")

    def _on_block_start(self, x):
        self._block_idx += 1
        self._norm_idx = 0
        self._block = {"block_idx": self._block_idx, "block_in": x[0].detach().cpu().tolist()}
        if self._labels is None:
            self._labels = [f"t{i}" for i in range(x.shape[1])]
        if self.verbose:
            print(f"  │   ┌── [TransformerBlock {self._block_idx}] 输入张量: (B={x.shape[0]}, T={x.shape[1]}, emb_dim={x.shape[2]})")

    def _on_layer_norm(self, **values):
        if self._block is not None:
            self._norm_idx += 1
            key = f"norm{self._norm_idx}"
            name = f"Pre-LN {self._norm_idx} ({key})"
            indent = "  │   │   "
            record = self._block[key] = {}
        else:
            name, indent = "Final LayerNorm (final_norm)", "  │   "
            record = self.record.setdefault("output_stage", {})["final_norm"] = {}
        _inspect_layer_norm(**values, verbose=self.verbose, show_data=self.show_data,
                            token_labels=self._labels, ln_name=name, indent=indent, record_dict=record)

    def _on_attention(self, **values):
        target = self._block if self._block is not None else self.record
        record = target["att"] = {}
        _inspect_attention(**values, verbose=self.verbose, show_data=self.show_data,
                           token_labels=self._labels, record_dict=record)

    def _on_feed_forward(self, **values):
        target = self._block if self._block is not None else self.record
        record = target["ffn"] = {}
        _inspect_feed_forward(**values, verbose=self.verbose, show_data=self.show_data,
                              token_labels=self._labels, record_dict=record)

    def _on_residual(self, name, shortcut, sublayer_out, out):
        if self.verbose:
            if name == "res1":
                print(f"  │   │   ├── [残差连接 1 (Shortcut 1)] x = shortcut + att_out: 形状保持 {tuple(out.shape)}")
                indent, sub_name = "  │   │   │     ", "att_out"
            else:
                print(f"  │   │   └── [残差连接 2 (Shortcut 2)] x = shortcut + ff_out: 最终块输出 {tuple(out.shape)}")
                indent, sub_name = "  │   │         ", "ff_out"
            if self.show_data:
                print_residual_comparison(shortcut[0], sublayer_out[0], out[0],
                                          indent=indent, row_labels=self._labels, sub_name=sub_name)
        self._block[name] = {
            "shortcut": shortcut[0].detach().cpu().tolist(),
            "sublayer_out": sublayer_out[0].detach().cpu().tolist(),
            "out": out[0].detach().cpu().tolist(),
        }

    def _on_block_end(self):
        if self.verbose:
            print(f"  │   └── [TransformerBlock {self._block_idx} 计算完成]\n  │")
        self.record.setdefault("blocks", []).append(self._block)
        self._block = None

    def _on_blocks_end(self, num_blocks):
        if self.verbose:
            print(f"  └── [Transformer 主干网络全部 {num_blocks} 个 Blocks 执行完毕]\n  │")
            print(f"  ┌── [3. 输出预测阶段 (Final LayerNorm & Output Head)]")

    def _on_model_end(self, **values):
        _inspect_output(**values, verbose=self.verbose, show_data=self.show_data,
                        token_labels=self._labels, record_dict=self.record["output_stage"])
        if self.verbose:
            print(f"  <<< GPTModel.forward end   | logits.shape={tuple(values['logits'].shape)}")

    def _on_generation_start(self, step, max_new_tokens, idx, idx_cond):
        if self.tokenizer is None:
            raise ValueError("TeachingTrace 的生成教学输出需要 tokenizer")
        self.record = _start_generation(self.tokenizer, step, max_new_tokens, idx, idx_cond, self.verbose)
        self._generating = True

    def _on_generation_end(self, step, logits_last, idx_next, idx):
        _finish_generation(self.tokenizer, step, logits_last, idx_next, idx, self.record, self.verbose)
        if self.steps_records is not None:
            self.steps_records.append(self.record)
        self._generating = False



def collect_and_print_model_info(model, cfg, tokenizer, print_weights=True, *, seed=None):
    """统计并打印模型超参规格、精确参数量统计及全部关键权重矩阵数据，同时返回结构化字典"""
    vocab_sorted = sorted(tokenizer.str_to_int.items(), key=lambda x: x[1])
    vocab_labels = [f"ID={idx} ('{word}')" for word, idx in vocab_sorted]
    pos_labels = [f"Pos={i}" for i in range(cfg["context_length"])]
    d = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    head_dim = d // n_heads

    # 从真实 PyTorch 模型提取精确参数计数
    tok_emb_p = sum(p.numel() for p in model.tok_emb.parameters())
    pos_emb_p = sum(p.numel() for p in model.pos_emb.parameters())
    b0 = model.trf_blocks[0] if model.trf_blocks else None
    ln1_p = sum(p.numel() for p in b0.norm1.parameters()) if b0 else 0
    qkv_p = sum(p.numel() for m in (b0.att.W_query, b0.att.W_key, b0.att.W_value) for p in m.parameters()) if b0 else 0
    out_proj_p = sum(p.numel() for p in b0.att.out_proj.parameters()) if b0 else 0
    mha_p = sum(p.numel() for p in b0.att.parameters()) if b0 else 0
    ln2_p = sum(p.numel() for p in b0.norm2.parameters()) if b0 else 0
    ffn_fc1_p = sum(p.numel() for p in b0.ff.layers[0].parameters()) if b0 else 0
    ffn_fc2_p = sum(p.numel() for p in b0.ff.layers[2].parameters()) if b0 else 0
    ffn_p = sum(p.numel() for p in b0.ff.parameters()) if b0 else 0
    block_p = sum(p.numel() for p in b0.parameters()) if b0 else 0
    all_blocks_p = sum(p.numel() for p in model.trf_blocks.parameters())
    final_ln_p = sum(p.numel() for p in model.final_norm.parameters())
    out_head_p = sum(p.numel() for p in model.out_head.parameters())
    total_p = sum(p.numel() for p in model.parameters())

    param_stats = {
        "tok_emb_p": tok_emb_p,
        "pos_emb_p": pos_emb_p,
        "ln1_p": ln1_p,
        "qkv_p": qkv_p,
        "out_proj_p": out_proj_p,
        "mha_p": mha_p,
        "ln2_p": ln2_p,
        "ffn_fc1_p": ffn_fc1_p,
        "ffn_fc2_p": ffn_fc2_p,
        "ffn_p": ffn_p,
        "block_p": block_p,
        "all_blocks_p": all_blocks_p,
        "final_ln_p": final_ln_p,
        "out_head_p": out_head_p,
        "total_p": total_p,
        "total_bytes_fp32": total_p * 4,
        "total_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }

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

    vocab_items = [f"'{w}': {idx}" for w, idx in vocab_sorted]
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
    qkv_formula = f"3 × ({d}×{d} + {d})" if cfg["qkv_bias"] else f"3 × {d}×{d}"
    print(f"        * 多头注意力 Q/K/V 权重 (qkv_bias={cfg['qkv_bias']}): {qkv_p:>4}  ({qkv_formula})")
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

    # 收集全量权重矩阵字典
    blocks_weights = []
    for i, block in enumerate(model.trf_blocks):
        fc1 = block.ff.layers[0]
        fc2 = block.ff.layers[2]
        blocks_weights.append({
            "block_idx": i,
            "norm1_scale": block.norm1.scale.detach().cpu().tolist(),
            "norm1_shift": block.norm1.shift.detach().cpu().tolist(),
            "W_query": block.att.W_query.weight.detach().cpu().tolist(),
            "W_query_bias": block.att.W_query.bias.detach().cpu().tolist() if block.att.W_query.bias is not None else None,
            "W_key": block.att.W_key.weight.detach().cpu().tolist(),
            "W_key_bias": block.att.W_key.bias.detach().cpu().tolist() if block.att.W_key.bias is not None else None,
            "W_value": block.att.W_value.weight.detach().cpu().tolist(),
            "W_value_bias": block.att.W_value.bias.detach().cpu().tolist() if block.att.W_value.bias is not None else None,
            "out_proj_weight": block.att.out_proj.weight.detach().cpu().tolist(),
            "out_proj_bias": block.att.out_proj.bias.detach().cpu().tolist(),
            "causal_mask": [[int(v) for v in row] for row in block.att.mask.detach().cpu().tolist()],
            "norm2_scale": block.norm2.scale.detach().cpu().tolist(),
            "norm2_shift": block.norm2.shift.detach().cpu().tolist(),
            "ff_fc1_weight": fc1.weight.detach().cpu().tolist(),
            "ff_fc1_bias": fc1.bias.detach().cpu().tolist(),
            "ff_fc2_weight": fc2.weight.detach().cpu().tolist(),
            "ff_fc2_bias": fc2.bias.detach().cpu().tolist(),
        })

    weights_dict = {
        "tok_emb_weight": model.tok_emb.weight.detach().cpu().tolist(),
        "pos_emb_weight": model.pos_emb.weight.detach().cpu().tolist(),
        "blocks": blocks_weights,
        "final_norm_scale": model.final_norm.scale.detach().cpu().tolist(),
        "final_norm_shift": model.final_norm.shift.detach().cpu().tolist(),
        "out_head_weight": model.out_head.weight.detach().cpu().tolist(),
    }

    if print_weights:
        print("\n" + "=" * 78)
        print("        GPT-2 Toy 全模块关键权重数据清单 (Initialized Weights)")
        seed_note = f"随机种子 torch.manual_seed({seed})" if seed is not None else "从当前模型读取权重"
        print(f"        ({seed_note}, PyTorch Linear 权重形状为 [out, in])")
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

            print(f"  │   ├── [norm1 (Pre-LayerNorm 1)]")
            print_1d_tensor(block.norm1.scale, indent="  │   │     ", label=f"norm1.scale {tuple(block.norm1.scale.shape)}")
            print_1d_tensor(block.norm1.shift, indent="  │   │     ", label=f"norm1.shift {tuple(block.norm1.shift.shape)}")

            print(f"  │   ├── [att (MultiHeadAttention) | 计算公式: x @ W^T (+ b)]")
            print(f"  │   │   ├── att.W_query.weight | 形状: {tuple(block.att.W_query.weight.shape)} (bias={block.att.W_query.bias is not None}):")
            print_2d_tensor(block.att.W_query.weight, indent="  │   │   │     ", row_labels=head_row_labels)

            print(f"  │   │   ├── att.W_key.weight   | 形状: {tuple(block.att.W_key.weight.shape)} (bias={block.att.W_key.bias is not None}):")
            print_2d_tensor(block.att.W_key.weight, indent="  │   │   │     ", row_labels=head_row_labels)

            print(f"  │   │   ├── att.W_value.weight | 形状: {tuple(block.att.W_value.weight.shape)} (bias={block.att.W_value.bias is not None}):")
            print_2d_tensor(block.att.W_value.weight, indent="  │   │   │     ", row_labels=head_row_labels)

            print(f"  │   │   ├── att.out_proj.weight | 形状: {tuple(block.att.out_proj.weight.shape)}:")
            print_2d_tensor(block.att.out_proj.weight, indent="  │   │   │     ", row_labels=emb_row_labels)
            print_1d_tensor(block.att.out_proj.bias, indent="  │   │   │     ", label=f"att.out_proj.bias {tuple(block.att.out_proj.bias.shape)}")

            print(f"  │   │   └── att.mask (因果掩码 Buffer, 非训练参数) | 形状: {tuple(block.att.mask.shape)} (1 表示遮蔽为 -inf):")
            mask_int_rows = [[int(v) for v in row] for row in block.att.mask.tolist()]
            for r_idx, m_row in enumerate(mask_int_rows):
                print(f"  │   │         Query Pos={r_idx}: {m_row}")

            print(f"  │   ├── [norm2 (Pre-LayerNorm 2)]")
            print_1d_tensor(block.norm2.scale, indent="  │   │     ", label=f"norm2.scale {tuple(block.norm2.scale.shape)}")
            print_1d_tensor(block.norm2.shift, indent="  │   │     ", label=f"norm2.shift {tuple(block.norm2.shift.shape)}")

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

    return param_stats, weights_dict


def _max_abs_diff(left, right):
    """比较嵌套数值列表；形状、非有限值或相反符号的无穷大不一致时返回 inf。"""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_max_abs_diff(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return math.inf
    if left == right:
        return 0.0
    if not math.isfinite(left) or not math.isfinite(right):
        return math.inf
    return abs(left - right)


@torch.no_grad()
def collect_same_token_example(model, tokenizer, text, positions=(2, 4)):
    """对固定重复词输入单独前向，返回与生成步骤无关的完整教学快照。"""
    token_ids = tokenizer.encode(text)
    if not token_ids or len(token_ids) > model.pos_emb.num_embeddings:
        raise ValueError("固定重复词输入必须非空，且长度不能超过模型上下文容量")
    if len(positions) != 2:
        raise ValueError("同词异位需要两个不同的位置")
    pos_a, pos_b = positions
    if pos_a == pos_b or min(pos_a, pos_b) < 0 or max(pos_a, pos_b) >= len(token_ids):
        raise ValueError("同词异位的位置必须不同且位于固定输入范围内")
    if token_ids[pos_a] != token_ids[pos_b]:
        raise ValueError("同词异位的两个位置必须具有相同的 Token ID")

    trace = TeachingTrace(tokenizer, verbose=False, show_data=False)
    ids = torch.tensor([token_ids], dtype=torch.long, device=model.tok_emb.weight.device)
    model(ids, trace=trace)
    record = trace.record
    embeddings = record["embeddings"]
    logits = record["output_stage"]["logits_all"]

    def prediction(position):
        token_id = max(range(len(logits[position])), key=logits[position].__getitem__)
        return {"id": token_id, "token": tokenizer.int_to_str.get(token_id, "<|unk|>")}

    return {
        "available": True,
        "reason": "",
        "source": "fixed_input",
        "input_text": text,
        "input_ids": token_ids,
        "input_tokens": [tokenizer.int_to_str.get(tid, "<|unk|>") for tid in token_ids],
        "token_labels": list(trace._labels),
        "pos_a": pos_a,
        "pos_b": pos_b,
        "token_id": token_ids[pos_a],
        "token_str": tokenizer.int_to_str.get(token_ids[pos_a], "<|unk|>"),
        "prediction_a": prediction(pos_a),
        "prediction_b": prediction(pos_b),
        "token_embeddings_equal": embeddings["tok_embeds"][pos_a] == embeddings["tok_embeds"][pos_b],
        "position_embeddings_equal": embeddings["pos_embeds"][pos_a] == embeddings["pos_embeds"][pos_b],
        "logits_equal": logits[pos_a] == logits[pos_b],
        "trace": record,
    }


def analyze_and_print_cross_step_insights(steps_records, same_token_example=None, *, tolerance=1e-6):
    """验证同位置前缀，并独立讲解固定输入的同词异位；所有结论来自实际数据。"""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("比较容差必须是非负有限数")
    checks = []
    invariance = {
        "status": "not_comparable", "reason": "至少需要两个生成步骤",
        "prefix_len": 0, "step_a": 1, "step_b": 2,
        "seq_len_a": 0, "seq_len_b": 0, "tolerance": tolerance,
    }
    if len(steps_records) >= 2:
        first, second = steps_records[:2]
        count = first["seq_len"]
        invariance.update({
            "prefix_len": count,
            "step_a": first["step_num"], "step_b": second["step_num"],
            "seq_len_a": count, "seq_len_b": second["seq_len"],
        })
        if first.get("training") or second.get("training"):
            invariance["reason"] = "训练模式不用于此项 KV Cache 教学验证；请使用 model.eval()"
        elif first.get("context_start", 0) != second.get("context_start", 0):
            invariance["reason"] = "上下文窗口已移动，历史 Token 的局部位置发生变化，不能直接对齐比较"
        elif not count or first["input_ids"] != second["input_ids"][:count]:
            invariance["reason"] = "两个步骤没有相同且完整保留的输入前缀"
        elif first["embeddings"]["pos_indices"] != second["embeddings"]["pos_indices"][:count]:
            invariance["reason"] = "两个步骤的前缀位置编号不同"
        elif len(first["blocks"]) != len(second["blocks"]):
            invariance["reason"] = "两个步骤的模型块数量不同"
        elif any(len(a["att"]["heads"]) != len(b["att"]["heads"])
                 for a, b in zip(first["blocks"], second["blocks"])):
            invariance["reason"] = "两个步骤的注意力头数量不同"
        else:
            def add(name, before, after):
                error = _max_abs_diff(before, after)
                checks.append({"name": name, "max_diff": error, "passed": error <= tolerance})

            add("输入嵌入 x = tok_emb + pos_emb",
                first["embeddings"]["x_emb"], second["embeddings"]["x_emb"][:count])
            for index, (before, after) in enumerate(zip(first["blocks"], second["blocks"])):
                prefix = f"Block {index} "
                add(prefix + "Pre-LN 1 输出", before["norm1"]["out"], after["norm1"]["out"][:count])
                for key, label in (("q_merged", "Queries"), ("k_merged", "Keys"), ("v_merged", "Values")):
                    add(prefix + f"MHA {label}", before["att"][key], after["att"][key][:count])
                for head, (left, right) in enumerate(zip(before["att"]["heads"], after["att"]["heads"])):
                    add(prefix + f"Head {head} 注意力权重前 {count}×{count} 子矩阵",
                        left["attn_weights"], [row[:count] for row in right["attn_weights"][:count]])
                for key, label in (("context_merged", "MHA 多头拼接"), ("out_proj_out", "MHA 输出投影")):
                    add(prefix + label, before["att"][key], after["att"][key][:count])
                for key, label in (("res1", "残差连接 1"), ("norm2", "Pre-LN 2"), ("res2", "残差连接 2")):
                    add(prefix + label, before[key]["out"], after[key]["out"][:count])
                add(prefix + "FFN 降维输出", before["ffn"]["fc2_out"], after["ffn"]["fc2_out"][:count])
            add("Final LayerNorm 输出", first["output_stage"]["final_norm"]["out"],
                second["output_stage"]["final_norm"]["out"][:count])
            add(f"词表预测 Logits 前 {count} 个位置", first["output_stage"]["logits_all"],
                second["output_stage"]["logits_all"][:count])
            passed = all(check["passed"] for check in checks)
            invariance["status"] = "verified" if passed else "different"
            invariance["reason"] = "所有已记录阶段均在容差内一致" if passed else "存在超过容差的实测差异"

    print("\n" + "=" * 78)
    print("       [深度洞察 1: 同位置前缀的跨步因果不变性验证]")
    print("=" * 78)
    if invariance["status"] == "not_comparable":
        print(f"  [跳过]: {invariance['reason']}")
    else:
        print(f"  对比 Step {invariance['step_a']} (T={invariance['seq_len_a']}) 与 "
              f"Step {invariance['step_b']} (T={invariance['seq_len_b']}) 的前 "
              f"{invariance['prefix_len']} 个位置，容差 = {tolerance:g}:")
        for check in checks:
            verdict = "通过 ✓" if check["passed"] else "未通过"
            print(f"   * {check['name']} : max|差值| = {check['max_diff']:.8g} ({verdict})")
        print(f"  [实测结论]: {invariance['reason']}。")
        if invariance["status"] == "verified":
            print("  在 eval、相同输入前缀与相同位置编号下，因果掩码允许复用历史 K/V。")
            print("  本例仍逐步重新计算整个窗口；窗口移动并重编号时需要重新检查缓存适用条件。")

    same = same_token_example if same_token_example is not None else {
        "available": False, "reason": "未提供独立的固定重复词输入",
    }
    print("\n" + "=" * 78)
    print("       [深度洞察 2: 固定重复词输入的同词异位对比]")
    print("=" * 78)
    if not same["available"]:
        print(f"  [跳过]: {same.get('reason', '重复词样本不可用')}")
    else:
        a, b = same["pos_a"], same["pos_b"]
        record = same["trace"]
        embeddings = record["embeddings"]
        print(f"  独立固定输入: {same['input_text']!r}，IDs = {same['input_ids']}")
        print(f"  对比 t{a} 与 t{b}，相同 Token ID={same['token_id']} ({same['token_str']!r})。")
        print(f"   * 词嵌入是否相同: {same['token_embeddings_equal']}")
        for position in (a, b):
            print(f"   * tok_emb[t{position}]: {format_1d(embeddings['tok_embeds'][position])}")
            print(f"   * pos_emb[Pos={position}]: {format_1d(embeddings['pos_embeds'][position])}")
            print(f"   * x_emb[t{position}]: {format_1d(embeddings['x_emb'][position])}")
        if record["blocks"]:
            for head in record["blocks"][0]["att"]["heads"]:
                for position in (a, b):
                    print(f"   * Head {head['head_idx']} t{position} 注意力权重: "
                          f"{format_1d(head['attn_weights'][position])}")
        for position, prediction in ((a, same["prediction_a"]), (b, same["prediction_b"])):
            print(f"   * t{position} 最终 Logits: "
                  f"{format_1d(record['output_stage']['logits_all'][position])}")
            print(f"     实测 Argmax -> ID={prediction['id']} ({prediction['token']!r})")
        position_result = "相同" if same["position_embeddings_equal"] else "不同"
        output_result = "相同" if same["logits_equal"] else "不同"
        print(f"  [实测结论]: 位置嵌入{position_result}，最终 Logits {output_result}。")
        print("  相同词嵌入会结合各自的位置向量和因果可见前缀；具体预测以上述实测值为准。")
    print("=" * 78)
    return {"invariance_checks": checks, "invariance": invariance, "same_token_diff_pos": same}
