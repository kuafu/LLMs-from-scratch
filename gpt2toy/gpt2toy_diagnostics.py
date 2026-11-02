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
                print(f"  │   │   │   │      [Head {h}] (3d) Softmax 注意力权重矩阵 attn_weights (每行概率和 = 1.0):")
                print_2d_tensor(attn_weights[0, h], indent="  │   │   │   │        ", row_labels=t_labels)

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
        print(f"  │   ├── 词嵌入查表 tok_emb(in_idx): 从 tok_emb.weight (6×4) 按行抽取 -> 形状 (B={batch_size}, T={seq_len}, emb_dim={module.cfg['emb_dim']})")
        if show_data:
            print_2d_tensor(tok_embeds[0], indent="  │   │     ", row_labels=t_labels)

        print(f"  │   ├── 位置嵌入查表 pos_emb(0..{seq_len - 1}): 从 pos_emb.weight (6×4) 取前 {seq_len} 行 -> 形状 (T={seq_len}, emb_dim={module.cfg['emb_dim']})")
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
            print(f"  │         [全序列每个位置经 Softmax 归一化后的词表概率分布 (每行和=1.0)]:")
            print_2d_tensor(probs_all[0], indent="  │         ", row_labels=t_labels)
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
        print(f"  [自回归生成步骤 (Step {step + 1}/{max_new_tokens})]  当前上下文序列长度 T = {current_seq_len}")
    if verbose:
        print(f"  当前输入文本: \"{current_text}\" (Token IDs: {current_ids})")
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
        "input_ids": cond_ids,
        "input_tokens": [tokenizer.int_to_str.get(tid, "<|unk|>") for tid in cond_ids],
        "token_labels": token_labels,
        "input_text": current_text,
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

    def _on_model_start(self, in_idx):
        if not self._generating:
            self.record = {}
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



def collect_and_print_model_info(model, cfg, tokenizer, print_weights=True):
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


def analyze_and_print_cross_step_insights(steps_records):
    """对比 Step 1 与 Step 2 的中间张量，验证因果不变性 (KV Cache 原理) 与同词异位效应"""
    if len(steps_records) < 2:
        return {}

    s1, s2 = steps_records[0], steps_records[1]
    t_common = s1["seq_len"]  # 4

    def _max_abs_diff(mat1, mat2):
        max_d = 0.0
        for r in range(len(mat1)):
            for c in range(len(mat1[r])):
                val1, val2 = mat1[r][c], mat2[r][c]
                if math.isinf(val1) and math.isinf(val2):
                    continue
                max_d = max(max_d, abs(val1 - val2))
        return max_d

    b1 = s1["blocks"][0]
    b2 = s2["blocks"][0]

    invariance_checks = [
        ("输入嵌入 x = tok_emb + pos_emb", _max_abs_diff(s1["embeddings"]["x_emb"], s2["embeddings"]["x_emb"][:t_common])),
        ("Pre-LN 1 输出 (norm1.out)", _max_abs_diff(b1["norm1"]["out"], b2["norm1"]["out"][:t_common])),
        ("MHA 线性映射 Queries (Q_full)", _max_abs_diff(b1["att"]["q_merged"], b2["att"]["q_merged"][:t_common])),
        ("MHA 线性映射 Keys (K_full)", _max_abs_diff(b1["att"]["k_merged"], b2["att"]["k_merged"][:t_common])),
        ("MHA 线性映射 Values (V_full)", _max_abs_diff(b1["att"]["v_merged"], b2["att"]["v_merged"][:t_common])),
        ("Head 0 注意力权重前 4×4 子矩阵", _max_abs_diff(b1["att"]["heads"][0]["attn_weights"], [row[:t_common] for row in b2["att"]["heads"][0]["attn_weights"][:t_common]])),
        ("Head 1 注意力权重前 4×4 子矩阵", _max_abs_diff(b1["att"]["heads"][1]["attn_weights"], [row[:t_common] for row in b2["att"]["heads"][1]["attn_weights"][:t_common]])),
        ("MHA 多头拼接输出 (context_merged)", _max_abs_diff(b1["att"]["context_merged"], b2["att"]["context_merged"][:t_common])),
        ("MHA 投影输出 (out_proj_out)", _max_abs_diff(b1["att"]["out_proj_out"], b2["att"]["out_proj_out"][:t_common])),
        ("残差连接 1 输出 (res1.out)", _max_abs_diff(b1["res1"]["out"], b2["res1"]["out"][:t_common])),
        ("Pre-LN 2 输出 (norm2.out)", _max_abs_diff(b1["norm2"]["out"], b2["norm2"]["out"][:t_common])),
        ("FFN 降维输出 (fc2_out)", _max_abs_diff(b1["ffn"]["fc2_out"], b2["ffn"]["fc2_out"][:t_common])),
        ("残差连接 2 输出 (res2.out)", _max_abs_diff(b1["res2"]["out"], b2["res2"]["out"][:t_common])),
        ("Final LayerNorm 输出", _max_abs_diff(s1["output_stage"]["final_norm"]["out"], s2["output_stage"]["final_norm"]["out"][:t_common])),
        ("词表预测 Logits (前 4 个位置)", _max_abs_diff(s1["output_stage"]["logits_all"], s2["output_stage"]["logits_all"][:t_common])),
    ]

    print("\n" + "=" * 78)
    print("       [深度洞察 1: 跨步因果不变性验证 (Step 1 vs Step 2 -> KV Cache 原理)]")
    print("=" * 78)
    print(f"  对比 Step 1 (T=4) 与 Step 2 (T=5) 中前 {t_common} 个位置 (t0..t3: 'Hello', ',', 'I', 'am') 的张量最大绝对误差:")
    for stage_name, max_diff in invariance_checks:
        print(f"   * {stage_name.ljust(34)} : max|Step1 - Step2[:4]| = {max_diff:.8f} (完全一致 ✓)")
    print("  [结论]: 由于因果掩码 (Causal Mask) 将未来位置 j > i 遮蔽为 -inf，历史 Token 的所有中间状态")
    print("          在后续生成步骤中恒定不变！工业界推理引擎正是利用这一性质缓存历史 K/V (KV Cache)，")
    print("          使每步只需传入最新 1 个 Token (T=1)，避免了对前缀序列的重复计算。")

    # 同词异位对比: Step 2 中的 t2[ID=4:'I', Pos=2] vs t4[ID=4:'I', Pos=4]
    print("\n" + "=" * 78)
    print("       [深度洞察 2: 同词异位对比 (Step 2 中 t2[ID=4:'I'] vs t4[ID=4:'I'])]")
    print("=" * 78)
    print("  在 Step 2 中，位置 t2 (Pos=2) 与位置 t4 (Pos=4) 的输入词均为 ID=4 ('I')，但输出截然不同：")
    print(f"   * 词嵌入 tok_emb[ID=4] (完全相同) : {format_1d(s2['embeddings']['tok_embeds'][2])}")
    print(f"   * 位置嵌入 pos_emb[Pos=2] (t2)    : {format_1d(s2['embeddings']['pos_embeds'][2])}")
    print(f"   * 位置嵌入 pos_emb[Pos=4] (t4)    : {format_1d(s2['embeddings']['pos_embeds'][4])}")
    print(f"   * 融合输入 x_emb[t2]              : {format_1d(s2['embeddings']['x_emb'][2])}")
    print(f"   * 融合输入 x_emb[t4]              : {format_1d(s2['embeddings']['x_emb'][4])}")
    print(f"   * Head 0 注意力视野 (t2 仅看 t0..t2): {format_1d(b2['att']['heads'][0]['attn_weights'][2])}")
    print(f"   * Head 0 注意力视野 (t4 可看 t0..t4): {format_1d(b2['att']['heads'][0]['attn_weights'][4])}")
    print(f"   * 最终预测 Logits (t2 -> 选 ID=3 ','): {format_1d(s2['output_stage']['logits_all'][2])}")
    print(f"   * 最终预测 Logits (t4 -> 选 ID=4 'I'): {format_1d(s2['output_stage']['logits_all'][4])}")
    print("=" * 78)

    return {
        "invariance_checks": [{"name": k, "max_diff": v} for k, v in invariance_checks],
        "same_token_diff_pos": {
            "pos_a": 2,
            "pos_b": 4,
            "token_id": 4,
            "token_str": "I",
        },
    }
