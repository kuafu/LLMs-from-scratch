# -*- coding: utf-8 -*-
"""
GPT-2 Toy 双轨剖析导出器 (Dual-Track Dissection Exporter)
生成：
  1. gpt2toy_dissection.md  —— 结构化图文 + 公式 + 完整矩阵对照 Markdown 深度剖析文档
  2. gpt2toy_inspector.html —— 零外部依赖的单文件交互式 HTML 观战面板

本模块只把诊断快照渲染为报告，不执行模型计算；维度、预测与比较结果均取自传入 payload。
"""

import os
import json
import math


def _f(val, p=4):
    """以固定精度格式化数值，并为正负无穷显示明确标记。"""
    if val is None:
        return "None"
    if isinstance(val, float) and math.isinf(val):
        return "-∞" if val < 0 else "+∞"
    return f"{val:.{p}f}"


def _vec_str(vec, p=4):
    """将数值向量逐项格式化为报告中的紧凑表示。"""
    return "[" + ", ".join(_f(v, p) for v in vec) + "]"


def _md_table(headers, rows, aligns=None):
    """用给定列对齐方式拼接 Markdown 表格，并转义各单元格内容。"""
    if not aligns:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    sep_map = {"left": ":---", "center": ":---:", "right": "---:"}
    lines = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "| " + " | ".join(sep_map.get(a, "---") for a in aligns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def _md_cell(value):
    """转义竖线并压平换行，避免 payload 文本破坏 Markdown 表格结构。"""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_text(value):
    """压平动态说明中的换行，保留 Token 等可见文本本身。"""
    return str(value).replace("\n", " ")


def generate_markdown_report(trace_data):
    """将诊断 payload 渲染为 Markdown；动态读取维度、权重、生成步骤与预测结果。"""
    cfg = trace_data["config"]
    ps = trace_data["param_stats"]
    w = trace_data["weights"]
    vocab = trace_data["vocab"]
    steps = trace_data["steps"]
    comp = trace_data["comparisons"]

    d = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    head_dim = cfg["head_dim"]
    ffn_dim = cfg["ffn_hidden_dim"]
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    vocab_map = {item["id"]: item["token"] for item in vocab}
    vocab_summary = ", ".join(f"{item['id']}: '{_md_text(item['token'])}'" for item in vocab)
    comparison_section = 4 + len(steps)
    head_ranges = ", ".join(
        f"Head {h} (列 {h * head_dim}..{(h + 1) * head_dim - 1})" for h in range(n_heads)
    )

    md = []
    md.append("# GPT-2 Toy 全链路像素级深度剖析报告 (Complete Dissection Report)\n")
    md.append(f"> **自动生成说明**：本报告由 `run_gpt2toy.py` 运行 `gpt2toy.py` 并录制生成，随机种子为 `{trace_data['seed']}`，处于推断模式 (`model.eval()`, `drop_rate={cfg['drop_rate']}`)。所有矩阵与中间张量数值均来自该运行的 PyTorch 前向传播。\n")
    md.append(f"**本次生成结果**：初始上下文 `\"{_md_text(trace_data['start_context'])}\"` (`{trace_data['initial_ids']}`) → 最终 Token IDs `{trace_data['final_ids']}`，文本 `\"{_md_text(trace_data['final_text'])}\"`。\n")

    # 目录
    md.append("## 目录 (Table of Contents)\n")
    md.append("1. [全局拓扑与张量维度演化图 (Architecture & Tensor Flow)](#1-全局拓扑与张量维度演化图)")
    md.append(f"2. [超参数规格与 {ps['total_p']} 个可学习参数精确拆解](#2-超参数规格与-{ps['total_p']}-个可学习参数精确拆解)")
    md.append("3. [全模块初始权重矩阵清单 (Initialized Weights)](#3-全模块初始权重矩阵清单)")
    for s_idx, step in enumerate(steps):
        pred = step["prediction"]
        section = 4 + s_idx
        md.append(f"{section}. [自回归 Step {step['step_num']} 全链路前向传播剖析 (`T={step['seq_len']}`: `{_md_text(step['input_text'])}` → `{_md_text(pred['next_token_str'])}`)](#{section}-自回归-step-{step['step_num']}-全链路前向传播剖析)")
    md.append(f"{comparison_section}. [跨步因果不变性验证 (KV Cache 核心原理) 与同词异位对比](#{comparison_section}-跨步因果不变性验证与同词异位对比)\n")

    # 1. Mermaid 流程图
    md.append("---\n## 1. 全局拓扑与张量维度演化图\n")
    md.append("```mermaid")
    md.append("flowchart TD")
    md.append(f'    In["输入序列 input_ids<br/>形状: (B=1, T)"] --> TokEmb["词嵌入查表 tok_emb<br/>weight: ({vocab_size}, {d}) → 输出: (1, T, {d})"]')
    md.append(f'    In --> PosEmb["位置嵌入查表 pos_emb<br/>weight: ({context_length}, {d}) → 输出: (T, {d})"]')
    md.append(f'    TokEmb --> AddEmb["广播逐元素相加: x = tok_emb + pos_emb<br/>形状: (1, T, {d})"]')
    md.append('    PosEmb --> AddEmb')
    md.append(f'    AddEmb --> LN1["Pre-LN 1: LayerNorm(x)<br/>(x - μ) / √(σ² + 1e-5) * γ + β → (1, T, {d})"]')
    md.append(f'    LN1 --> QKV["Q / K / V 线性投影 (bias={cfg["qkv_bias"]})<br/>x_norm1 @ W_q^T, W_k^T, W_v^T → 3 × (1, T, {d})"]')
    md.append(f'    QKV --> Split["多头拆分 (view + transpose)<br/>{head_ranges} → 3 × (1, {n_heads}, T, {head_dim})"]')
    md.append(f'    Split --> Scores["四步注意力得分演化<br/>① Q@K^T → ② /√{head_dim} → ③ 上三角置 -∞ → ④ Softmax → (1, {n_heads}, T, T)"]')
    md.append(f'    Scores --> Context["Value 加权求和 & 多头拼接<br/>attn_weights @ V → (1, {n_heads}, T, {head_dim}) → Concat (1, T, {d})"]')
    md.append(f'    Context --> OutProj["MHA 输出线性投影 out_proj<br/>context @ W_out^T + b_out → (1, T, {d})"]')
    md.append(f'    AddEmb -- "Shortcut 1 (残差主干)" --> Res1["残差连接 1: x = shortcut1 + att_out<br/>形状: (1, T, {d})"]')
    md.append('    OutProj --> Res1')
    md.append(f'    Res1 --> LN2["Pre-LN 2: LayerNorm(x)<br/>形状: (1, T, {d})"]')
    md.append(f'    LN2 --> FFN1["FFN 升维层: Linear({d} → {ffn_dim})<br/>形状: (1, T, {ffn_dim})"]')
    md.append(f'    FFN1 --> GELU["非线性激活: GELU(h1)<br/>形状: (1, T, {ffn_dim})"]')
    md.append(f'    GELU --> FFN2["FFN 降维层: Linear({ffn_dim} → {d})<br/>形状: (1, T, {d})"]')
    md.append(f'    Res1 -- "Shortcut 2 (残差主干)" --> Res2["残差连接 2: x = shortcut2 + ff_out<br/>形状: (1, T, {d})"]')
    md.append('    FFN2 --> Res2')
    md.append(f'    Res2 --> FinalLN["最终层归一化: final_norm(x)<br/>形状: (1, T, {d})"]')
    md.append(f'    FinalLN --> OutHead["词表投影头: out_head(x)<br/>x_final @ W_head^T → Logits (1, T, {vocab_size})"]')
    md.append('    OutHead --> NextTok["提取末尾时间步 logits[:, -1, :]<br/>Softmax 概率分布 → Argmax 采样下一个 Token"]')
    md.append("```\n")

    # 2. 超参数与参数统计
    md.append(f"---\n## 2. 超参数规格与 {ps['total_p']} 个可学习参数精确拆解\n")
    md.append("### 2.1 模型超参数表与微型词表\n")
    cfg_rows = [
        ["`vocab_size` (词表大小)", f"`{vocab_size}`", f"词表 ID 与 Token：{vocab_summary}"],
        ["`context_length` (最大上下文长度)", f"`{context_length}`", f"位置嵌入表支持 `Pos = 0..{context_length - 1}`"],
        ["`emb_dim` (嵌入维度 $d_{\\text{model}}$)", f"`{d}`", f"主干残差流向量宽度为 {d} 维"],
        ["`n_heads` (注意力头数 $H$)", f"`{n_heads}`", f"每个头维度 $d_k = d_v = {d} / {n_heads} = {head_dim}$"],
        ["`ffn_hidden_dim` (FFN 隐藏维)", f"`{ffn_dim}`", f"前馈网络映射维度为 {d} → {ffn_dim} → {d}"],
        ["`n_layers` (Transformer 层数)", f"`{cfg['n_layers']}`", f"包含 {cfg['n_layers']} 个 `TransformerBlock`"],
        ["`qkv_bias` / `drop_rate`", f"`{cfg['qkv_bias']}` / `{cfg['drop_rate']}`", f"Q/K/V bias={cfg['qkv_bias']}；推断模式下 Dropout 关闭"],
    ]
    md.append(_md_table(["超参数名称", "取值", "架构意义说明"], cfg_rows, ["left", "center", "left"]) + "\n")

    md.append("### 2.2 全模块可学习参数量实测分布\n")
    param_rows = [
        ["1. 输入嵌入阶段", "`tok_emb.weight`", f"`({cfg['vocab_size']}, {d})`", "`False`", f"`{ps['tok_emb_p']}`"],
        ["1. 输入嵌入阶段", "`pos_emb.weight`", f"`({cfg['context_length']}, {d})`", "`False`", f"`{ps['pos_emb_p']}`"],
        ["2. TransformerBlock (每层)", "`norm1` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['ln1_p']}`"],
        ["2. TransformerBlock (每层)", "`att.W_query / W_key / W_value`", f"`3 × ({d}, {d})`", f"`{cfg['qkv_bias']}`", f"`{ps['qkv_p']}`"],
        ["2. TransformerBlock (每层)", "`att.out_proj` (`weight` + `bias`)", f"`({d}, {d}) + ({d},)`", "`True`", f"`{ps['out_proj_p']}`"],
        ["2. TransformerBlock (每层)", "`norm2` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['ln2_p']}`"],
        ["2. TransformerBlock (每层)", f"`ff.layers[0]` (升维 `{d}→{ffn_dim}`)", f"`({ffn_dim}, {d}) + ({ffn_dim},)`", "`True`", f"`{ps['ffn_fc1_p']}`"],
        ["2. TransformerBlock (每层)", f"`ff.layers[2]` (降维 `{ffn_dim}→{d}`)", f"`({d}, {ffn_dim}) + ({d},)`", "`True`", f"`{ps['ffn_fc2_p']}`"],
        [f"**所有 TransformerBlock (共 {cfg['n_layers']} 层)**", "**每层参数总和**", "—", "—", f"**`{ps['all_blocks_p']}`**"],
        ["3. 输出预测阶段", "`final_norm` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['final_ln_p']}`"],
        ["3. 输出预测阶段", "`out_head.weight` (词表投影)", f"`({cfg['vocab_size']}, {d})`", "`False`", f"`{ps['out_head_p']}`"],
        ["**全模型总计 (Total)**", "**所有可学习参数**", f"**Float32 占 {ps['total_bytes_fp32']} Bytes**", "—", f"**`{ps['total_p']}`**"],
    ]
    md.append(_md_table(["所属阶段", "模块 / 参数张量", "权重形状", "含 Bias", "参数量"], param_rows, ["left", "left", "center", "center", "right"]) + "\n")

    # 3. 全模块初始权重矩阵清单
    md.append("---\n## 3. 全模块初始权重矩阵清单 (Initialized Weights)\n")
    md.append("> **注意**：PyTorch 中 `nn.Linear(in_features, out_features)` 的 `.weight` 存储形状为 `(out_features, in_features)`，前向计算公式为 $y = x W^T + b$。\n")

    md.append(f"### 3.1 词嵌入表 `tok_emb.weight` `({vocab_size} × {d})` 与 位置嵌入表 `pos_emb.weight` `({context_length} × {d})`\n")
    tok_rows = []
    for idx, row in enumerate(w["tok_emb_weight"]):
        tok_rows.append([f"`ID={idx}` (`'{vocab_map[idx]}'`)", *[_f(v) for v in row]])
    md.append("**词嵌入查找表 `tok_emb.weight` (行号即 Token ID)：**\n")
    md.append(_md_table(["Token ID (词元)", *[f"Dim {i}" for i in range(d)]], tok_rows) + "\n")

    pos_rows = []
    for idx, row in enumerate(w["pos_emb_weight"]):
        pos_rows.append([f"`Pos={idx}`", *[_f(v) for v in row]])
    md.append(f"**位置嵌入查找表 `pos_emb.weight` (行号即序列位置 `0..{context_length - 1}`)：**\n")
    md.append(_md_table(["位置索引 Pos", *[f"Dim {i}" for i in range(d)]], pos_rows) + "\n")

    for block_idx, bw in enumerate(w["blocks"]):
        md.append(f"### 3.2 `TransformerBlock {block_idx}` 内部权重矩阵\n")
        md.append(f"- **`norm1` (Pre-LN 1)**: `scale` = `{_vec_str(bw['norm1_scale'])}`, `shift` = `{_vec_str(bw['norm1_shift'])}`\n")
        md.append(f"- **`norm2` (Pre-LN 2)**: `scale` = `{_vec_str(bw['norm2_scale'])}`, `shift` = `{_vec_str(bw['norm2_shift'])}`\n")

        for w_title, w_key in [
            (f"Query 投影矩阵 `att.W_query.weight` `({d} × {d})`", "W_query"),
            (f"Key 投影矩阵 `att.W_key.weight` `({d} × {d})`", "W_key"),
            (f"Value 投影矩阵 `att.W_value.weight` `({d} × {d})`", "W_value"),
        ]:
            md.append(f"**{w_title}**（每 {head_dim} 行归属一个注意力头，共 {n_heads} 个头）：\n")
            qkv_rows = []
            for r_idx, row in enumerate(bw[w_key]):
                h_id = r_idx // head_dim
                d_id = r_idx % head_dim
                qkv_rows.append([f"`OutDim {r_idx}` (`Head {h_id}, d{d_id}`)", *[_f(v) for v in row]])
            md.append(_md_table(["输出维度 (归属头)", *[f"InDim {i}" for i in range(d)]], qkv_rows) + "\n")

        md.append(f"**多头输出投影层 `att.out_proj.weight` `({d} × {d})` 与 `bias` `({d},)`：**\n")
        op_rows = []
        for r_idx, row in enumerate(bw["out_proj_weight"]):
            op_rows.append([f"`OutDim {r_idx}`", *[_f(v) for v in row], _f(bw["out_proj_bias"][r_idx])])
        md.append(_md_table(["输出维度", *[f"InDim {i} (H{i // head_dim},d{i % head_dim})" for i in range(d)], "Bias"], op_rows) + "\n")

        md.append(f"<details>\n<summary><b>点击展开查看 FFN 升维层 ({ffn_dim}×{d}) 与 降维层 ({d}×{ffn_dim}) 完整权重表</b></summary>\n\n")
        fc1_rows = []
        for r_idx, row in enumerate(bw["ff_fc1_weight"]):
            fc1_rows.append([f"`HidDim {r_idx:02d}`", *[_f(v) for v in row], _f(bw["ff_fc1_bias"][r_idx])])
        md.append(f"**FFN 升维层 `ff.layers[0].weight` `({ffn_dim} × {d})` 与 `bias` `({ffn_dim},)`：**\n\n")
        md.append(_md_table(["隐藏维索引", *[f"InDim {i}" for i in range(d)], "Bias"], fc1_rows) + "\n\n")

        fc2_rows = []
        split_at = (ffn_dim + 1) // 2
        for r_idx, row in enumerate(bw["ff_fc2_weight"]):
            fc2_rows.append([f"`OutDim {r_idx}`", _vec_str(row[:split_at]), _vec_str(row[split_at:]), _f(bw["ff_fc2_bias"][r_idx])])
        md.append(f"**FFN 降维层 `ff.layers[2].weight` `({d} × {ffn_dim})` 与 `bias` `({d},)`：**\n\n")
        md.append(_md_table(["输出维度", f"InDim [0..{split_at - 1}]", f"InDim [{split_at}..{ffn_dim - 1}]", "Bias"], fc2_rows, ["left", "left", "left", "right"]) + "\n")
        md.append("</details>\n")

    md.append(f"### 3.3 最终归一化 `final_norm` 与 词表输出头 `out_head.weight` `({vocab_size} × {d})`\n")
    md.append(f"- **`final_norm`**: `scale` = `{_vec_str(w['final_norm_scale'])}`, `shift` = `{_vec_str(w['final_norm_shift'])}`\n")
    oh_rows = []
    for idx, row in enumerate(w["out_head_weight"]):
        oh_rows.append([f"`ID={idx}` (`'{vocab_map[idx]}'`)", *[_f(v) for v in row]])
    md.append(_md_table(["预测目标词表项", *[f"InDim {i}" for i in range(d)]], oh_rows) + "\n")

    # 4 & 5. 分步前向传播详细剖析
    for s_idx, step in enumerate(steps):
        sec_num = 4 + s_idx
        step_num = step["step_num"]
        T = step["seq_len"]
        t_labels = step["token_labels"]
        in_ids = step["input_ids"]
        emb = step["embeddings"]
        blk = step["blocks"][0]
        out_st = step["output_stage"]
        pred = step["prediction"]

        md.append(f"---\n## {sec_num}. 自回归 Step {step_num} 全链路前向传播剖析\n")
        md.append(f"- **当前步骤输入文本**: `\"{_md_text(step['input_text'])}\"`\n")
        md.append(f"- **输入 Token IDs (`input_ids`)**: `{in_ids}` (形状 `(B=1, T={T})`)\n")
        md.append(f"- **位置编号 (`pos_indices`)**: `{emb['pos_indices']}`\n")

        # 阶段 1: 嵌入查表与逐元素相加
        md.append(f"### {sec_num}.1 阶段一：输入嵌入查表与广播相加 (`Input Embeddings`)\n")
        md.append("计算公式：\n$$X_{t, d} = W_{\\text{tok}}[\\text{in\\_idx}_t, d] + W_{\\text{pos}}[t, d]$$\n")
        emb_rows = []
        for t in range(T):
            tid = in_ids[t]
            emb_rows.append([
                f"`{t_labels[t]}`",
                f"`tok_emb[{tid}]` = `{_vec_str(emb['tok_embeds'][t])}`",
                f"`pos_emb[{t}]` = `{_vec_str(emb['pos_embeds'][t])}`",
                f"`{_vec_str(emb['x_emb'][t])}`",
            ])
        md.append(_md_table(
            ["序列位置 Token", "词嵌入查表 `tok_emb[ID]`", "位置嵌入查表 `pos_emb[Pos]`", "融合结果 `x = tok + pos`"],
            emb_rows, ["left", "left", "left", "left"]
        ) + "\n")

        # 阶段 2: Pre-LN 1
        md.append(f"### {sec_num}.2 阶段二：`Pre-LN 1 (norm1)` 内部统计量与归一化验算\n")
        md.append(f"对每个 Token 的 {d} 维向量独立求均值 $\\mu_t = \\frac{{1}}{{{d}}}\\sum_{{d=0}}^{{{d - 1}}} x_{{t,d}}$ 与有偏方差 $\\sigma^2_t = \\frac{{1}}{{{d}}}\\sum_{{d=0}}^{{{d - 1}}} (x_{{t,d}}-\\mu_t)^2$，再执行标准化 $\\hat{{x}}_{{t,d}} = \\frac{{x_{{t,d}} - \\mu_t}}{{\\sqrt{{\\sigma^2_t + 10^{{-5}}}}}}$：\n")
        ln1 = blk["norm1"]
        ln1_rows = []
        for t in range(T):
            ln1_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(ln1['in'][t])}`",
                _f(ln1["mean"][t]),
                _f(ln1["var"][t]),
                _f(ln1["std"][t]),
                f"`{_vec_str(ln1['out'][t])}`",
                f"`μ'={_f(ln1['post_mean'][t])}, σ'²={_f(ln1['post_var'][t])}`",
            ])
        md.append(_md_table(
            ["Token", "输入向量 `x`", "均值 `μ`", "有偏方差 `σ²`", "分母 `√(σ²+ε)`", "归一化输出 `x_norm1`", "后验校验"],
            ln1_rows, ["left", "left", "right", "right", "right", "left", "center"]
        ) + "\n")

        # 阶段 3: MultiHeadAttention
        att = blk["att"]
        md.append(f"### {sec_num}.3 阶段三：因果多头注意力 (`MultiHeadAttention`) 全链路显微镜\n")
        md.append(f"#### ({sec_num}.3.1) 切头前完整 {d} 维线性映射 vs {n_heads} 个注意力头\n")
        md.append(f"线性变换 $Q = X_{{\\text{{norm1}}}} W_Q^T,\\; K = X_{{\\text{{norm1}}}} W_K^T,\\; V = X_{{\\text{{norm1}}}} W_V^T$ 输出形状均为 `(1, T, {d})`。随后重排为 `(1, {n_heads}, T, {head_dim})`，各头对应切片：{head_ranges}。\n")
        qkv_split_rows = []
        for t in range(T):
            qkv_split_rows.append([f"`{t_labels[t]}`", f"`{_vec_str(att['q_merged'][t])}`"] + [
                f"`{_vec_str(h_obj[key][t])}`"
                for key in ("q", "k", "v") for h_obj in att["heads"]
            ])
        md.append(_md_table(
            ["Token", f"`Q_full ({d}维)`"] + [f"`{key.upper()}_Head{h}`" for key in ("q", "k", "v") for h in range(n_heads)],
            qkv_split_rows, ["left"] * (2 + 3 * n_heads)
        ) + "\n")

        md.append(f"#### ({sec_num}.3.2) 注意力得分矩阵四步演化 (`Raw` → `Scaled` → `Masked` → `Softmax`)\n")
        md.append(f"缩放因子 $\\sqrt{{d_k}} = \\sqrt{{{head_dim}}} \\approx {math.sqrt(head_dim):.4f}$。因果掩码将上三角区域 ($j > i$，即未来位置) 置为 $-\\infty$，使得 $\\text{{softmax}}$ 后未来位置权重严格为 `0.0000`：\n")

        col_headers = ["Query 位置 \\ Key 位置"] + [f"`t{j}` (`'{step['input_tokens'][j]}'`)" for j in range(T)]
        for h_obj in att["heads"]:
            h_idx = h_obj["head_idx"]
            md.append(f"**【Head {h_idx}】四步注意力矩阵 (`{T} × {T}`)：**\n")

            for stage_title, mat_key in [
                (f"① 原始点积得分 `raw_scores = Q_{h_idx} @ K_{h_idx}^T`", "raw_scores"),
                (f"② 缩放得分 `unmasked_scaled = raw_scores / √{head_dim}`", "unmasked_scaled_scores"),
                (f"③ 因果掩码遮蔽后得分 `masked_scores` ($j > i$ 置为 $-\\infty$)", "masked_scores"),
                ("④ Softmax 注意力权重 `attn_weights`（推断模式关闭 Dropout，行和为 `1.0000`）", "attn_weights"),
            ]:
                md.append(f"- **{stage_title}**：\n")
                m_rows = []
                for i_r in range(T):
                    m_rows.append([f"`{t_labels[i_r]}`", *[_f(v) for v in h_obj[mat_key][i_r]]])
                md.append(_md_table(col_headers, m_rows) + "\n")

        md.append(f"#### ({sec_num}.3.3) Value 加权求和、多头左右拼接与 `out_proj` 输出线性投影\n")
        md.append(f"每个头独立计算 $\\text{{context}}_h = \\text{{attn\\_weights}}_h V_h \\in \\mathbb{{R}}^{{T \\times {head_dim}}}$，再将 {n_heads} 个头拼接为 $\\text{{context}}_{{\\text{{merged}}}} \\in \\mathbb{{R}}^{{T \\times {d}}}$，最后经 `out_proj` 映射为 `att_out`：\n")
        ctx_rows = []
        for t in range(T):
            ctx_rows.append([f"`{t_labels[t]}`"] + [
                f"`{_vec_str(h_obj['context_head'][t])}`" for h_obj in att["heads"]
            ] + [f"`{_vec_str(att['context_merged'][t])}`", f"`{_vec_str(att['out_proj_out'][t])}`"])
        md.append(_md_table(
            ["Token"] + [f"`Head {h} Context ({head_dim}维)`" for h in range(n_heads)] + [f"多头拼接 `Concat ({d}维)`", f"投影输出 `att_out ({d}维)`"],
            ctx_rows, ["left"] * (n_heads + 3)
        ) + "\n")

        # 阶段 4: 残差连接 1
        md.append(f"### {sec_num}.4 阶段四：残差连接 1 (`Shortcut 1`: `x = shortcut1 + att_out`)\n")
        res1 = blk["res1"]
        r1_rows = []
        for t in range(T):
            r1_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(res1['shortcut'][t])}`",
                f"`{_vec_str(res1['sublayer_out'][t])}`",
                f"`{_vec_str(res1['out'][t])}`",
            ])
        md.append(_md_table(["Token", "主干输入 `shortcut1`", "子层增量 `att_out`", "残差融合输出 `x_res1`"], r1_rows, ["left"] * 4) + "\n")

        # 阶段 5: Pre-LN 2 & FFN & 残差连接 2
        md.append(f"### {sec_num}.5 阶段五：`Pre-LN 2 (norm2)` → 前馈网络 (`FFN`) → 残差连接 2 (`Shortcut 2`)\n")
        ln2 = blk["norm2"]
        ln2_rows = []
        for t in range(T):
            ln2_rows.append([
                f"`{t_labels[t]}`",
                _f(ln2["mean"][t]),
                _f(ln2["var"][t]),
                _f(ln2["std"][t]),
                f"`{_vec_str(ln2['out'][t])}`",
            ])
        md.append("**(1) `Pre-LN 2 (norm2)` 统计量与归一化输出：**\n")
        md.append(_md_table(["Token", "均值 `μ`", "方差 `σ²`", "标准差 `std`", "归一化输出 `x_norm2`"], ln2_rows, ["left", "right", "right", "right", "left"]) + "\n")

        ffn = blk["ffn"]
        md.append(f"**(2) `FFN` 升维 (`{d}→{ffn_dim}`)、GELU 激活与降维 (`{ffn_dim}→{d}`) 输出：**\n")
        ffn_rows = []
        for t in range(T):
            ffn_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(ffn['fc1_out'][t][:min(4, ffn_dim)])} ...` (共{ffn_dim}维)",
                f"`{_vec_str(ffn['gelu_out'][t][:min(4, ffn_dim)])} ...` (共{ffn_dim}维)",
                f"`{_vec_str(ffn['fc2_out'][t])}`",
            ])
        md.append(_md_table(["Token", f"升维 `fc1_out` (前{min(4, ffn_dim)}维预览)", f"激活 `GELU(fc1)` (前{min(4, ffn_dim)}维预览)", f"降维输出 `ff_out ({d}维)`"], ffn_rows, ["left"] * 4) + "\n")

        res2 = blk["res2"]
        md.append("**(3) 残差连接 2 (`x_block_out = shortcut2 + ff_out`) 三列对照：**\n")
        r2_rows = []
        for t in range(T):
            r2_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(res2['shortcut'][t])}`",
                f"`{_vec_str(res2['sublayer_out'][t])}`",
                f"`{_vec_str(res2['out'][t])}`",
            ])
        md.append(_md_table(["Token", "主干输入 `shortcut2 (x_res1)`", "子层增量 `ff_out`", "Block 最终输出 `x_block_out`"], r2_rows, ["left"] * 4) + "\n")

        # 阶段 6: Final LayerNorm & Out Head & 采样决策
        md.append(f"### {sec_num}.6 阶段六：最终层归一化 (`final_norm`)、词表 Logits 投影与采样决策\n")
        fln = out_st["final_norm"]
        fln_rows = []
        for t in range(T):
            fln_rows.append([
                f"`{t_labels[t]}`",
                _f(fln["mean"][t]),
                _f(fln["var"][t]),
                f"`{_vec_str(fln['out'][t])}`",
                f"`{_vec_str(out_st['logits_all'][t])}`",
            ])
        md.append(_md_table(["Token", "均值 `μ`", "方差 `σ²`", f"`final_norm` 输出 `({d}维)`", f"词表 Logits `out_head(x)` `({vocab_size}维)`"], fln_rows, ["left", "right", "right", "left", "left"]) + "\n")

        md.append(f"**末尾时间步 `{pred['last_token_label']}` 对 {vocab_size} 个候选词的预测概率分布与 Argmax 采样：**\n")
        cand_rows = []
        for c in pred["candidates"]:
            mark = "**✅ 选中 (Argmax)**" if c["is_selected"] else ""
            cand_rows.append([
                f"`ID={c['id']}`",
                f"`'{c['token']}'`",
                _f(c["logit"]),
                f"`{c['prob'] * 100:.2f}%`",
                mark,
            ])
        md.append(_md_table(["候选 Token ID", "词元字符串", "原始 Logit", "Softmax 概率", "采样决策"], cand_rows, ["center", "left", "right", "right", "center"]) + "\n")
        md.append(f"> **Step {step_num} 结论**：贪婪采样选中 `ID={pred['next_token_id']}` (`\"{_md_text(pred['next_token_str'])}\"`), 序列更新为 `{pred['updated_ids']}` → 文本：`\"{_md_text(pred['updated_text'])}\"`。\n")

    # 6. Cross-step invariance and independent same-token comparison
    md.append(f"---\n## {comparison_section}. 跨步因果不变性验证与同词异位对比\n")
    inv = comp.get("invariance", {})
    checks = comp.get("invariance_checks", [])
    tol = inv.get("tolerance", 0.0)
    checks_pass = bool(checks) and all(
        item.get("passed") is True and item.get("max_diff", math.inf) <= tol
        for item in checks
    )
    inv_verified = inv.get("status") == "verified" and checks_pass
    status_names = {"verified": "已验证", "different": "存在差异", "not_comparable": "不可比较"}
    inv_status = inv.get("status", "not_comparable")
    md.append(f"### {comparison_section}.1 跨步因果不变性检查（{status_names.get(inv_status, _md_text(str(inv_status)))}）\n")
    if inv_verified:
        md.append(f"比较 Step {inv['step_a']} (T={inv['seq_len_a']}) 与 Step {inv['step_b']} (T={inv['seq_len_b']}) 的前 {inv['prefix_len']} 个相同前缀位置；按 tolerance `{tol:g}` 判断数值差异。\n")
    else:
        md.append(f"本次无法确认可复用的相同前缀位置。原因：{_md_text(inv.get('reason', '比较数据不足。'))}。\n")
    inv_rows = []
    for item in checks:
        passed = item.get("passed") is True and item.get("max_diff", math.inf) <= tol
        status_text = "通过 tolerance 检查 ✅" if inv_verified and passed else ("超出 tolerance" if not passed else "单项通过；整体条件未确认")
        inv_rows.append([item["name"], f"`{item['max_diff']:.8f}`", status_text])
    if inv_rows:
        md.append(_md_table(["前向传播中间张量阶段", "最大绝对误差", "检查结果"], inv_rows, ["left", "right", "center"]) + "\n")
    else:
        md.append("当前没有可列出的逐张量检查。\n")
    if inv_verified:
        md.append(f"> **KV Cache 原理**：两次前向传播使用相同前缀和相同位置索引、并在推断模式关闭 Dropout 时，因果遮罩让每个位置只依赖可见前缀（含当前位置）；本次已将 K/V 和其他列出的张量按 tolerance `{tol:g}` 比较。KV Cache 可在这些条件下复用历史 K/V。不同硬件或计算路径可能产生微小浮点差异，不保证逐位完全相同。\n")
    else:
        md.append("> **KV Cache 条件**：因果遮罩只排除未来 Token；复用历史 K/V 还要求两次计算使用相同前缀和位置索引、推断模式关闭 Dropout，并按指定 tolerance 比较。本次数据未通过完整检查，因此不能作为 KV Cache 验证成功的证据。\n")

    same = comp.get("same_token_diff_pos", {})
    md.append(f"### {comparison_section}.2 同一 Token 出现在不同位置的独立前向传播\n")
    if not same.get("available", False):
        md.append(f"此比较不可用：{_md_text(same.get('reason', '未提供比较记录。'))}\n")
    else:
        same_trace = same["trace"]
        same_emb = same_trace["embeddings"]
        same_block = same_trace["blocks"][0]
        pos_a, pos_b = same["pos_a"], same["pos_b"]
        same_out = same_trace["output_stage"]
        pred_a, pred_b = same["prediction_a"], same["prediction_b"]
        md.append(f"比较输入来自固定、独立的示例 `{_md_text(same['input_text'])}`（IDs `{same['input_ids']}`），不是上方自回归步骤之一。Token `{_md_text(same['token_str'])}` (ID={same['token_id']}) 位于 Pos={pos_a} 和 Pos={pos_b}。\n")
        head_idx = 0
        head = same_block["att"]["heads"][head_idx]
        if same["logits_equal"]:
            prediction_note = "logits_equal=true，两个位置的 Logits 相同；以下仍列出各自 argmax。"
        else:
            prediction_note = "logits_equal=false，两个位置的 Logits 不同；argmax 是否相同以列出的实际 Token 为准。"
        token_emb_text = "相同" if same["token_embeddings_equal"] else "不同"
        pos_emb_text = "相同" if same["position_embeddings_equal"] else "不同"
        logits_text = "相同" if same["logits_equal"] else "不同"
        diff_pos_rows = [
            ["词嵌入查表", f"`{_vec_str(same_emb['tok_embeds'][pos_a])}`", f"`{_vec_str(same_emb['tok_embeds'][pos_b])}`", f"词嵌入{token_emb_text}（token_embeddings_equal={same['token_embeddings_equal']}）"],
            ["位置嵌入", f"`{_vec_str(same_emb['pos_embeds'][pos_a])}` (Pos={pos_a})", f"`{_vec_str(same_emb['pos_embeds'][pos_b])}` (Pos={pos_b})", f"位置嵌入{pos_emb_text}（position_embeddings_equal={same['position_embeddings_equal']}）"],
            ["融合输入 `x_emb`", f"`{_vec_str(same_emb['x_emb'][pos_a])}`", f"`{_vec_str(same_emb['x_emb'][pos_b])}`", "词嵌入与位置嵌入相加"],
            [f"Head {head_idx} 注意力权重（示例）", f"`{_vec_str(head['attn_weights'][pos_a])}`", f"`{_vec_str(head['attn_weights'][pos_b])}`", "每个位置的因果可见前缀长度不同"],
            ["Block 输出 `res2.out`", f"`{_vec_str(same_block['res2']['out'][pos_a])}`", f"`{_vec_str(same_block['res2']['out'][pos_b])}`", "首个 TransformerBlock 的实测输出"],
            [f"Logits `({vocab_size} 维)`", f"`{_vec_str(same_out['logits_all'][pos_a])}`", f"`{_vec_str(same_out['logits_all'][pos_b])}`", f"Logits {logits_text}（logits_equal={same['logits_equal']}）；argmax: ID={pred_a['id']} ('{pred_a['token']}') vs ID={pred_b['id']} ('{pred_b['token']}')；{prediction_note}"],
        ]
        md.append(_md_table(["比较阶段", f"Pos {pos_a} / token `{same['token_str']}`", f"Pos {pos_b} / token `{same['token_str']}`", "实测解读"], diff_pos_rows, ["left"] * 4) + "\n")

    return "\n".join(md)


def generate_html_inspector(trace_data):
    """将实际诊断 payload 嵌入单文件交互页面；页面不重新运行模型。

    维度和预测均从 payload 驱动；非有限浮点数先转成可解析标记，内联 JSON
    再转义尖括号与 ampersand，避免内容意外闭合 script 元素。
    """
    # 将 JSON 中可能出现的 -inf 转换为特殊字符串以便前端 JS 安全解析并渲染
    def _sanitize_for_json(obj):
        """递归把 NaN/Infinity 转为 JSON 可表达的字符串，其余数据原样保留。"""
        if isinstance(obj, float):
            if math.isinf(obj):
                return "-Infinity" if obj < 0 else "Infinity"
            if math.isnan(obj):
                return "NaN"
            return obj
        elif isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    clean_data = _sanitize_for_json(trace_data)
    # Prevent payload text such as an EOS token or a literal </script> from
    # terminating the embedded JSON script element.
    json_str = json.dumps(clean_data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPT-2 Toy 全链路交互式观战面板 (Interactive Dissection Inspector)</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-alt: #263349;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-soft: rgba(56, 189, 248, 0.15);
    --green: #4ade80;
    --green-soft: rgba(74, 222, 128, 0.16);
    --amber: #fbbf24;
    --amber-soft: rgba(251, 191, 36, 0.16);
    --purple: #c084fc;
    --purple-soft: rgba(192, 132, 252, 0.16);
    --rose: #fb7185;
    --mono: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
    line-height: 1.55;
    padding: 20px 28px 60px;
  }
  .header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
  }
  .header h1 {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .header .sub {
    color: var(--muted);
    font-size: 13.5px;
    margin-top: 6px;
  }
  .badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .badge {
    font-family: var(--mono);
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--accent);
  }
  .badge.green { color: var(--green); border-color: rgba(74,222,128,0.4); background: var(--green-soft); }
  .badge.amber { color: var(--amber); border-color: rgba(251,191,36,0.4); background: var(--amber-soft); }
  .badge.purple { color: var(--purple); border-color: rgba(192,132,252,0.4); background: var(--purple-soft); }

  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
    background: var(--panel);
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid var(--border);
    position: sticky;
    top: 10px;
    z-index: 100;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
  }
  .tabs {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13.5px;
    font-weight: 600;
    transition: all 0.15s;
  }
  .tab-btn:hover {
    color: var(--text);
    background: rgba(255,255,255,0.05);
  }
  .tab-btn.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-color: rgba(56, 189, 248, 0.45);
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  .act-btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 7px;
    font-size: 12.5px;
    cursor: pointer;
  }
  .act-btn:hover { border-color: var(--accent); color: var(--accent); }

  .flow-banner {
    background: rgba(56, 189, 248, 0.07);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  .token-Zero {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--panel);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 13px;
    margin-right: 6px;
  }
  .token-Zero .pos { color: var(--amber); font-size: 11px; }
  .token-Zero .str { color: var(--green); font-weight: 700; }

  .stage-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .stage-header {
    padding: 13px 18px;
    background: rgba(255,255,255,0.025);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
  }
  .stage-header:hover { background: rgba(255,255,255,0.05); }
  .stage-title {
    font-weight: 700;
    font-size: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .stage-body {
    padding: 18px;
  }
  .stage-card.collapsed .stage-body { display: none; }
  .stage-card.collapsed .stage-header { border-bottom: none; }

  .formula-box {
    background: #0b1120;
    border-left: 3px solid var(--accent);
    padding: 10px 14px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12.5px;
    color: #cbd5e1;
    margin-bottom: 14px;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 16px;
  }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: 14px;
  }

  .sub-box {
    background: #131c2e;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 13px;
  }
  .sub-box h4 {
    font-size: 13.5px;
    color: var(--accent);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  table.mat-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 12.5px;
  }
  table.mat-table th, table.mat-table td {
    border: 1px solid #283548;
    padding: 6px 9px;
    text-align: right;
  }
  table.mat-table th {
    background: #1a2639;
    color: var(--muted);
    font-weight: 600;
    text-align: center;
  }
  table.mat-table td.row-lbl {
    text-align: left;
    background: #172133;
    color: #e2e8f0;
    font-weight: 600;
    white-space: nowrap;
  }
  table.mat-table td.masked-cell {
    color: #64748b;
    background: rgba(15, 23, 42, 0.7);
    font-style: italic;
  }
  td[data-tip] {
    position: relative;
    cursor: help;
  }
  td[data-tip]:hover {
    outline: 1.5px solid var(--amber);
    z-index: 5;
  }

  .prob-bar-wrap {
    width: 140px;
    background: #0f172a;
    height: 16px;
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }
  .prob-bar {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
  }
  .prob-bar.best {
    background: linear-gradient(90deg, #22c55e, #4ade80);
  }
  .selected-row td {
    background: rgba(74, 222, 128, 0.12) !important;
  }

  #tooltip {
    position: fixed;
    pointer-events: none;
    background: #090d16;
    border: 1px solid var(--amber);
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 12px;
    z-index: 9999;
    display: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    max-width: 480px;
    white-space: pre-wrap;
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🔬 GPT-2 Toy 全链路交互式观战面板 (Pixel-Level Dissection Inspector)</h1>
    <div class="sub">
      从词表嵌入、位置嵌入、LayerNorm 统计量、多头注意力四步演化、残差连接到自回归采样决策与 KV Cache 因果不变性验证（悬停单元格可查看验算公式）
    </div>
  </div>
  <div class="badge-row" id="header-badges"></div>
</div>

<div class="toolbar">
  <div class="tabs" id="main-tabs">
    <button class="tab-btn" onclick="switchTab('compare')">⚡ 跨步对比 (KV Cache 原理 & 同词异位)</button>
    <button class="tab-btn" onclick="switchTab('weights')">📦 模型架构与初始权重矩阵</button>
  </div>
  <div class="actions">
    <button class="act-btn" onclick="toggleAllStages(false)">展开全部模块</button>
    <button class="act-btn" onclick="toggleAllStages(true)">折叠全部模块</button>
  </div>
</div>

<div id="view-container"></div>
<div id="tooltip"></div>

<script>
const TRACE = __TRACE_JSON__;
const CFG = TRACE.config;
const EMB_DIM = CFG.emb_dim;
const N_HEADS = CFG.n_heads;
const HEAD_DIM = CFG.head_dim;
const FFN_DIM = CFG.ffn_hidden_dim;

function escHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmt(v, p=4) {
  if (v === null || v === undefined) return "None";
  if (v === "-Infinity") return "-∞";
  if (v === "Infinity") return "+∞";
  if (typeof v === "number") return v.toFixed(p);
  return String(v);
}

function fmtVec(arr, p=4) {
  return "[" + arr.map(x => fmt(x, p)).join(", ") + "]";
}

function renderHeaderBadges() {
  const c = TRACE.config;
  const p = TRACE.param_stats;
  const el = document.getElementById("header-badges");
  el.innerHTML = `
    <span class="badge">Seed: ${TRACE.seed}</span>
    <span class="badge">vocab_size: ${c.vocab_size}</span>
    <span class="badge">context_length: ${c.context_length}</span>
    <span class="badge">emb_dim: ${c.emb_dim}</span>
    <span class="badge purple">n_heads: ${c.n_heads} (head_dim=${c.head_dim})</span>
    <span class="badge amber">FFN: ${c.emb_dim}→${c.ffn_hidden_dim}→${c.emb_dim}</span>
    <span class="badge green">Total Params: ${p.total_p} (${p.total_bytes_fp32} B)</span>
    <span class="badge">Final: ${escHtml(TRACE.final_text)}</span>
  `;
}

function renderTabButtons() {
  const root = document.getElementById("main-tabs");
  const stepButtons = TRACE.steps.map((step, i) =>
    `<button class="tab-btn${i === 0 ? ' active' : ''}" onclick="switchTab('step${i}')">▶ Step ${step.step_num} 观战 (T=${step.seq_len}: "${escHtml(step.input_text)}" → "${escHtml(step.prediction.next_token_str)}")</button>`
  ).join("");
  root.insertAdjacentHTML("afterbegin", stepButtons);
}

function toggleStage(headerEl) {
  headerEl.parentElement.classList.toggle("collapsed");
}

function toggleAllStages(collapse) {
  document.querySelectorAll(".stage-card").forEach(card => {
    if (collapse) card.classList.add("collapsed");
    else card.classList.remove("collapsed");
  });
}

function heatBg(val, minV=-1.0, maxV=1.0, isProb=false) {
  if (val === "-Infinity" || val === "Infinity" || typeof val !== "number") return "";
  if (isProb) {
    const alpha = Math.max(0.04, Math.min(0.65, val * 0.75));
    return `background: rgba(56, 189, 248, ${alpha.toFixed(3)});`;
  }
  if (val >= 0) {
    const ratio = Math.min(1, val / (maxV || 1));
    return `background: rgba(74, 222, 128, ${(ratio * 0.35).toFixed(3)});`;
  } else {
    const ratio = Math.min(1, Math.abs(val) / Math.abs(minV || 1));
    return `background: rgba(251, 113, 133, ${(ratio * 0.35).toFixed(3)});`;
  }
}

function renderStepView(stepIdx) {
  const step = TRACE.steps[stepIdx];
  const T = step.seq_len;
  const tLabels = step.token_labels;
  const emb = step.embeddings;
  const blk = step.blocks[0];
  const att = blk.att;
  const ln1 = blk.norm1;
  const ln2 = blk.norm2;
  const ffn = blk.ffn;
  const res1 = blk.res1;
  const res2 = blk.res2;
  const outSt = step.output_stage;
  const pred = step.prediction;

  let html = "";

  // 顶部序列横幅
  const tokPills = step.input_ids.map((tid, i) =>
    `<span class="token-Zero"><span class="pos">Pos=${i}</span><span>ID=${tid}</span><span class="str">'${escHtml(step.input_tokens[i])}'</span></span>`
  ).join(" → ");

  html += `
    <div class="flow-banner">
      <div>
        <div style="font-size:12.5px;color:var(--muted);margin-bottom:6px;">
          自回归生成 Step ${step.step_num} 输入上下文序列 (B=1, T=${T})：
        </div>
        <div>${tokPills}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:12px;color:var(--muted);">本步贪婪采样预测结果</div>
        <div style="font-family:var(--mono);font-size:15px;color:var(--green);font-weight:700;">
          next_token = ID=${pred.next_token_id} ("${escHtml(pred.next_token_str)}") → "${escHtml(pred.updated_text)}"
        </div>
      </div>
    </div>
  `;

  // Stage 1: Input Embeddings
  let embRows = "";
  for (let t = 0; t < T; t++) {
    const tid = step.input_ids[t];
    let cells = "";
    for (let d = 0; d < EMB_DIM; d++) {
      const tv = emb.tok_embeds[t][d];
      const pv = emb.pos_embeds[t][d];
      const xv = emb.x_emb[t][d];
      const tip = `位置 t${t} ('${step.input_tokens[t]}'), 维度 d${d}:\\ntok_emb[ID=${tid}, d${d}] (${fmt(tv)}) + pos_emb[Pos=${t}, d${d}] (${fmt(pv)}) = ${fmt(xv)}`;
      cells += `<td data-tip="${escHtml(tip)}" style="${heatBg(xv, -3, 3)}">${fmt(xv)}</td>`;
    }
    embRows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        <td style="text-align:left;color:#cbd5e1;">tok[${tid}] = ${fmtVec(emb.tok_embeds[t])}</td>
        <td style="text-align:left;color:#cbd5e1;">pos[${t}] = ${fmtVec(emb.pos_embeds[t])}</td>
        ${cells}
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>1️⃣ 阶段一：输入词嵌入与位置嵌入广播相加 (Input Embeddings)</span>
          <span class="badge">(1, ${T}) → tok_emb (1, ${T}, ${EMB_DIM}) + pos_emb (${T}, ${EMB_DIM}) = x (1, ${T}, ${EMB_DIM})</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          数学公式: X[t, d] = tok_emb.weight[input_ids[t], d] + pos_emb.weight[t, d]  (将鼠标悬停在右侧结果单元格上可查看逐元素相加验算)
        </div>
        <table class="mat-table">
          <thead>
            <tr>
              <th>序列位置 Token</th>
              <th>词嵌入查表 tok_emb[ID] (1×${EMB_DIM})</th>
              <th>位置嵌入查表 pos_emb[Pos] (1×${EMB_DIM})</th>
              ${Array.from({length: EMB_DIM}, (_, d) => `<th>x[d${d}]</th>`).join("")}
            </tr>
          </thead>
          <tbody>${embRows}</tbody>
        </table>
      </div>
    </div>
  `;

  // Stage 2: Pre-LN 1
  let ln1Rows = "";
  for (let t = 0; t < T; t++) {
    const mu = ln1.mean[t];
    const vr = ln1.var[t];
    const st = ln1.std[t];
    let outCells = "";
    for (let d = 0; d < EMB_DIM; d++) {
      const xin = ln1.in[t][d];
      const xout = ln1.out[t][d];
      const tip = `Pre-LN 1 验算 (t${t}, d${d}):\\n(x=${fmt(xin)} - μ=${fmt(mu)}) / √(σ²=${fmt(vr)} + 1e-5) = ${fmt(xout)}`;
      outCells += `<td data-tip="${escHtml(tip)}" style="${heatBg(xout, -2, 2)}">${fmt(xout)}</td>`;
    }
    ln1Rows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        <td style="text-align:left;color:var(--muted);">${fmtVec(ln1.in[t])}</td>
        <td style="color:var(--amber);">${fmt(mu)}</td>
        <td style="color:var(--amber);">${fmt(vr)}</td>
        <td style="color:var(--amber);">${fmt(st)}</td>
        ${outCells}
        <td style="color:var(--green);text-align:center;">μ'=${fmt(ln1.post_mean[t])}, σ'²=${fmt(ln1.post_var[t])}</td>
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>2️⃣ 阶段二：Pre-LayerNorm 1 内部统计量与标准化 (norm1)</span>
          <span class="badge">x (1, ${T}, ${EMB_DIM}) → x_norm1 (1, ${T}, ${EMB_DIM})</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          数学公式: μ_t = mean(x_t),  σ²_t = var(x_t, unbiased=False),  x_norm1 = (x_t - μ_t) / √(σ²_t + 1e-5) * scale + shift
        </div>
        <table class="mat-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>输入向量 x</th>
              <th>均值 μ</th>
              <th>有偏方差 σ²</th>
              <th>标准差 √(σ²+ε)</th>
              ${Array.from({length: EMB_DIM}, (_, d) => `<th>norm1[d${d}]</th>`).join("")}
              <th>归一化后验校验</th>
            </tr>
          </thead>
          <tbody>${ln1Rows}</tbody>
        </table>
      </div>
    </div>
  `;

  // Stage 3: Multi-Head Attention
  let qkvSplitRows = "";
  for (let t = 0; t < T; t++) {
    qkvSplitRows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        <td style="text-align:left;">${fmtVec(att.q_merged[t])}</td>
        ${att.heads.map((head, h) => `<td style="text-align:left;color:${h % 2 === 0 ? 'var(--accent)' : 'var(--purple)'};">${fmtVec(head.q[t])}</td>`).join("")}
        ${att.heads.map((head, h) => `<td style="text-align:left;color:${h % 2 === 0 ? 'var(--accent)' : 'var(--purple)'};">${fmtVec(head.k[t])}</td>`).join("")}
        ${att.heads.map((head, h) => `<td style="text-align:left;color:${h % 2 === 0 ? 'var(--accent)' : 'var(--purple)'};">${fmtVec(head.v[t])}</td>`).join("")}
      </tr>
    `;
  }

  let headsEvoHtml = "";
  for (let h = 0; h < att.heads.length; h++) {
    const hObj = att.heads[h];
    const stages = [
      { title: `① 原始点积 Q_${h} @ K_${h}^T`, key: "raw_scores", isProb: false },
      { title: `② 缩放得分 (/ √${HEAD_DIM})`, key: "unmasked_scaled_scores", isProb: false },
      { title: `③ 因果掩码遮蔽 (j>i 置 -∞)`, key: "masked_scores", isProb: false },
      { title: "④ Softmax 注意力权重（推断模式关闭 Dropout）", key: "attn_weights", isProb: true },
    ];

    let stageBoxes = "";
    stages.forEach(st => {
      let ths = `<th>Q \\ K</th>` + step.input_tokens.map((tok, j) => `<th>t${j}('${escHtml(tok)}')</th>`).join("");
      let trs = "";
      for (let i = 0; i < T; i++) {
        let tds = `<td class="row-lbl">t${i}('${escHtml(step.input_tokens[i])}')</td>`;
        for (let j = 0; j < T; j++) {
          const val = hObj[st.key][i][j];
          const isMasked = val === "-Infinity";
          const qVec = hObj.q[i];
          const kVec = hObj.k[j];
          const dotStr = qVec.map((v, d) => `${fmt(v)}×${fmt(kVec[d])}`).join(" + ") + ` = ${fmt(hObj.raw_scores[i][j])}`;
          const tip = `Head ${h} [Query t${i}, Key t${j}]:\\nQ_${h}[t${i}] = ${fmtVec(qVec)}, K_${h}[t${j}] = ${fmtVec(kVec)}\\n原始点积 = ${dotStr}\\n缩放后 = ${fmt(hObj.unmasked_scaled_scores[i][j])}\\nSoftmax权重 = ${fmt(hObj.attn_weights[i][j])}`;
          tds += `<td class="${isMasked ? 'masked-cell' : ''}" data-tip="${escHtml(tip)}" style="${heatBg(val, -0.6, 0.6, st.isProb)}">${fmt(val)}</td>`;
        }
        trs += `<tr>${tds}</tr>`;
      }
      stageBoxes += `
        <div class="sub-box">
          <h4><span>[Head ${h}] ${st.title}</span><span class="badge">${T}×${T}</span></h4>
          <table class="mat-table">
            <thead><tr>${ths}</tr></thead>
            <tbody>${trs}</tbody>
          </table>
        </div>
      `;
    });

    headsEvoHtml += `
      <div style="margin-top:16px;">
        <div style="font-weight:700;font-size:14px;margin-bottom:10px;color:${h===0 ? 'var(--accent)' : 'var(--purple)'};">
          🔹 Head ${h} 注意力矩阵四步同屏演化 (源自线性映射列 [${h*HEAD_DIM}..${(h+1)*HEAD_DIM-1}])：
        </div>
        <div class="grid-4">${stageBoxes}</div>
      </div>
    `;
  }

  let ctxRows = "";
  for (let t = 0; t < T; t++) {
    ctxRows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        ${att.heads.map((head, h) => `<td style="text-align:left;color:${h % 2 === 0 ? 'var(--accent)' : 'var(--purple)'};">${fmtVec(head.context_head[t])}</td>`).join("")}
        <td style="text-align:left;">${fmtVec(att.context_merged[t])}</td>
        <td style="text-align:left;color:var(--green);font-weight:600;">${fmtVec(att.out_proj_out[t])}</td>
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>3️⃣ 阶段三：因果多头注意力机制全链路显微镜 (Multi-Head Attention)</span>
          <span class="badge">(1, ${T}, ${EMB_DIM}) → ${N_HEADS} heads × (${T}, ${HEAD_DIM}) → Attn (${T}×${T}) → Concat & out_proj (1, ${T}, ${EMB_DIM})</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          Attention(Q, K, V) = softmax( mask(Q @ K^T / √${HEAD_DIM}) ) @ V  |  每个 Head 对应 ${HEAD_DIM} 列
        </div>
        <div class="sub-box" style="margin-bottom:14px;">
          <h4><span>(3.1) 切头前完整 ${EMB_DIM} 维 Q_full 与多头空间拆分后的 Q / K / V</span><span class="badge">(1, ${T}, ${EMB_DIM}) → (1, ${N_HEADS}, ${T}, ${HEAD_DIM})</span></h4>
          <table class="mat-table">
            <thead>
              <tr>
                <th>Token</th>
                <th>Q_full (切头前 ${EMB_DIM}维)</th>
                ${att.heads.map((head, h) => `<th>Head ${h}: Q_${h}</th>`).join("")}
                ${att.heads.map((head, h) => `<th>Head ${h}: K_${h}</th>`).join("")}
                ${att.heads.map((head, h) => `<th>Head ${h}: V_${h}</th>`).join("")}
              </tr>
            </thead>
            <tbody>${qkvSplitRows}</tbody>
          </table>
        </div>
        ${headsEvoHtml}
        <div class="sub-box" style="margin-top:16px;">
          <h4><span>(3.3) Value 加权求和 (attn_weights @ V)、多头左右拼接与 out_proj 线性映射</span><span class="badge">2×(${T}, 2) → (${T}, 4) → (${T}, 4)</span></h4>
          <table class="mat-table">
            <thead>
              <tr>
                <th>Token</th>
                ${att.heads.map((head, h) => `<th>Head ${h}: attn_${h} @ V_${h} (${HEAD_DIM}维)</th>`).join("")}
                <th>拼接 context_vec (${EMB_DIM}维)</th>
                <th>输出线性投影 att_out = out_proj(context)</th>
              </tr>
            </thead>
            <tbody>${ctxRows}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // Stage 4: Residual 1
  let res1Rows = "";
  for (let t = 0; t < T; t++) {
    res1Rows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        <td style="text-align:left;">${fmtVec(res1.shortcut[t])}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(res1.sublayer_out[t])}</td>
        <td style="text-align:left;color:var(--green);font-weight:600;">${fmtVec(res1.out[t])}</td>
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>4️⃣ 阶段四：残差连接 1 (Shortcut 1: x_res1 = shortcut1 + att_out)</span>
          <span class="badge">(1, ${T}, ${EMB_DIM}) + (1, ${T}, ${EMB_DIM}) → (1, ${T}, ${EMB_DIM})</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <table class="mat-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>主干输入 shortcut1 (即 x_emb)</th>
              <th>MHA 子层增量 att_out</th>
              <th>残差相加输出 x_res1 = shortcut1 + att_out</th>
            </tr>
          </thead>
          <tbody>${res1Rows}</tbody>
        </table>
      </div>
    </div>
  `;

  // Stage 5: Pre-LN 2 + FFN + Residual 2
  let ffnRows = "";
  for (let t = 0; t < T; t++) {
    ffnRows += `
      <tr>
        <td class="row-lbl">${escHtml(tLabels[t])}</td>
        <td style="text-align:left;">${fmtVec(ln2.out[t])} (μ=${fmt(ln2.mean[t])}, σ²=${fmt(ln2.var[t])})</td>
        <td style="text-align:left;color:var(--muted);font-size:11.5px;">${fmtVec(ffn.fc1_out[t].slice(0, Math.min(6, FFN_DIM)))} ... (${FFN_DIM}维)</td>
        <td style="text-align:left;color:var(--amber);font-size:11.5px;">${fmtVec(ffn.gelu_out[t].slice(0, Math.min(6, FFN_DIM)))} ... (${FFN_DIM}维)</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(ffn.fc2_out[t])}</td>
        <td style="text-align:left;color:var(--green);font-weight:600;">${fmtVec(res2.out[t])}</td>
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>5️⃣ 阶段五：Pre-LN 2 (norm2) → 逐位置前馈网络 (FFN ${EMB_DIM}→${FFN_DIM}→${EMB_DIM}) → 残差连接 2</span>
          <span class="badge">(1, ${T}, ${EMB_DIM}) → (1, ${T}, ${FFN_DIM}) → (1, ${T}, ${EMB_DIM})</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          FFN(x_norm2) = Linear_${FFN_DIM}→${EMB_DIM}( GELU( Linear_${EMB_DIM}→${FFN_DIM}(x_norm2) ) )  |  x_block_out = x_res1 + ff_out
        </div>
        <table class="mat-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>Pre-LN 2 输出 x_norm2</th>
              <th>升维 Linear(${EMB_DIM}→${FFN_DIM}) 前${Math.min(6, FFN_DIM)}维</th>
              <th>非线性激活 GELU(h1) 前${Math.min(6, FFN_DIM)}维</th>
              <th>降维输出 ff_out (${EMB_DIM}维)</th>
              <th>残差连接 2 输出 (x_res1 + ff_out)</th>
            </tr>
          </thead>
          <tbody>${ffnRows}</tbody>
        </table>
      </div>
    </div>
  `;

  // Stage 6: Final Norm & Output Logits & Next Token Prediction
  const fln = outSt.final_norm;
  let logitRows = "";
  for (let t = 0; t < T; t++) {
    const isLast = (t === T - 1);
    let lCells = "";
    for (let v = 0; v < TRACE.config.vocab_size; v++) {
      const lv = outSt.logits_all[t][v];
      const pv = outSt.probs_all[t][v];
      const tip = `位置 t${t} 预测词 ID=${v} ('${TRACE.vocab[v].token}'):\\nLogit = ${fmt(lv)}\\nSoftmax 概率 = ${(pv*100).toFixed(2)}%`;
      lCells += `<td data-tip="${escHtml(tip)}" style="${heatBg(pv, 0, 1, true)}">${fmt(lv)} <span style="color:var(--muted);font-size:11px;">(${(pv*100).toFixed(1)}%)</span></td>`;
    }
    logitRows += `
      <tr class="${isLast ? 'selected-row' : ''}">
        <td class="row-lbl">${escHtml(tLabels[t])} ${isLast ? '⬅ 末尾预测步' : ''}</td>
        <td style="text-align:left;">${fmtVec(fln.out[t])}</td>
        ${lCells}
      </tr>
    `;
  }

  let candRows = "";
  pred.candidates.forEach(c => {
    const pct = (c.prob * 100).toFixed(2);
    candRows += `
      <tr class="${c.is_selected ? 'selected-row' : ''}">
        <td class="row-lbl">ID=${c.id} ('${escHtml(c.token)}')</td>
        <td>${fmt(c.logit)}</td>
        <td>${pct}%</td>
        <td style="text-align:left;">
          <div class="prob-bar-wrap">
            <div class="prob-bar ${c.is_selected ? 'best' : ''}" style="width:${Math.max(2, c.prob*100)}%"></div>
          </div>
          ${c.is_selected ? '<span style="color:var(--green);font-weight:700;margin-left:8px;">✔ Argmax 选中</span>' : ''}
        </td>
      </tr>
    `;
  });

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>6️⃣ 阶段六：最终层归一化 (final_norm)、词表投影 (out_head) 与 下一个词采样决策</span>
          <span class="badge">(1, ${T}, ${EMB_DIM}) → Logits (1, ${T}, ${CFG.vocab_size}) → Argmax: ID=${pred.next_token_id} ('${escHtml(pred.next_token_str)}')</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="grid-2">
          <div class="sub-box">
            <h4><span>全序列每个位置的 Final LayerNorm 与 ${CFG.vocab_size} 词表 Logits (括号内为概率)</span></h4>
            <table class="mat-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>final_norm(x) (${EMB_DIM}维)</th>
                  ${TRACE.vocab.map(v => `<th>ID=${v.id}<br/>'${escHtml(v.token)}'</th>`).join("")}
                </tr>
              </thead>
              <tbody>${logitRows}</tbody>
            </table>
          </div>
          <div class="sub-box">
            <h4><span>末尾时间步 ${escHtml(pred.last_token_label)} 候选词概率分布与贪婪采样</span></h4>
            <table class="mat-table">
              <thead>
                <tr>
                  <th>候选 Token</th>
                  <th>Logit 得分</th>
                  <th>Softmax 概率</th>
                  <th>概率分布可视化 & 决策</th>
                </tr>
              </thead>
              <tbody>${candRows}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `;

  return html;
}

function renderCompareView() {
  const comparisons = TRACE.comparisons || {};
  const inv = comparisons.invariance || {};
  const checks = comparisons.invariance_checks || [];
  const tolerance = Number(inv.tolerance || 0);
  const checksPass = checks.length > 0 && checks.every(c => c.passed === true && Number(c.max_diff) <= tolerance);
  const verified = inv.status === "verified" && checksPass;
  const checkRows = checks.map(c => {
    const passed = c.passed === true && Number(c.max_diff) <= tolerance;
    const status = verified && passed ? "通过 tolerance 检查" : (passed ? "单项通过；整体条件未确认" : "未通过 tolerance 检查");
    return "<tr><td class=\\"row-lbl\\">" + escHtml(c.name) + "</td><td>" + Number(c.max_diff).toFixed(8) + "</td><td>" + status + "</td></tr>";
  }).join("");

  let kvContent = "";
  if (verified) {
    const stepA = TRACE.steps.find(s => s.step_num === inv.step_a);
    const stepB = TRACE.steps.find(s => s.step_num === inv.step_b);
    if (stepA && stepB) {
      const blockA = stepA.blocks[0];
      const blockB = stepB.blocks[0];
      const prefixLen = Math.min(inv.prefix_len, stepA.seq_len, stepB.seq_len);
      const rows = Array.from({length: prefixLen}, (_, t) =>
        "<tr><td class=\\"row-lbl\\">" + escHtml(stepA.token_labels[t]) + "</td>" +
        "<td style=\\"text-align:left;color:var(--accent);\\">" + fmtVec(blockA.att.k_merged[t]) + "</td>" +
        "<td style=\\"text-align:left;color:var(--accent);\\">" + fmtVec(blockB.att.k_merged[t]) + "</td>" +
        "<td style=\\"text-align:left;color:var(--purple);\\">" + fmtVec(blockA.att.v_merged[t]) + "</td>" +
        "<td style=\\"text-align:left;color:var(--purple);\\">" + fmtVec(blockB.att.v_merged[t]) + "</td></tr>"
      ).join("");
      kvContent = "<div class=\\"sub-box\\"><h4><span>Step " + stepA.step_num + " 与 Step " + stepB.step_num + " 的 K / V 实测对照</span></h4>" +
        "<table class=\\"mat-table\\"><thead><tr><th>相同前缀位置</th><th>Step " + stepA.step_num + " K (T=" + stepA.seq_len + ")</th><th>Step " + stepB.step_num + " K (T=" + stepB.seq_len + ")</th><th>Step " + stepA.step_num + " V</th><th>Step " + stepB.step_num + " V</th></tr></thead><tbody>" + rows + "</tbody></table></div>";
    }
  }
  const invReason = inv.reason || (inv.status === "verified" && !checksPass ? "逐张量检查缺失或未通过 tolerance，不能标记为已验证。" : "没有符合条件的比较资料。");
  const invText = verified
    ? "比较 Step " + inv.step_a + " (T=" + inv.seq_len_a + ") 与 Step " + inv.step_b + " (T=" + inv.seq_len_b + ") 的 " + inv.prefix_len + " 个相同前缀位置；tolerance=" + tolerance + "。"
    : "未确认可复用的相同前缀。状态：" + escHtml(inv.status || "not_comparable") + "。原因：" + escHtml(invReason);
  const kvExplanation = verified
    ? "因果遮罩排除未来 Token。当两次前向传播使用相同前缀与位置索引、推断模式关闭 Dropout 时，可以按 tolerance " + tolerance + " 比较并复用历史 K/V。本次检查通过；不同硬件或核心仍可能产生微小浮点差异。"
    : "因果遮罩只排除未来 Token。复用历史 K/V 还要求相同前缀与位置索引、推断模式关闭 Dropout，并按 tolerance 检查；这份记录没有通过完整条件，不能当作 KV Cache 成功证明。";

  const same = comparisons.same_token_diff_pos || {available: false, reason: "没有提供同 Token 不同位置比较。"};
  let sameContent;
  if (!same.available) {
    sameContent = "<div class=\\"formula-box\\">此比较不可用：" + escHtml(same.reason || "未提供比较记录。") + "</div>";
  } else {
    const trace = same.trace;
    const emb = trace.embeddings;
    const block = trace.blocks[0];
    const out = trace.output_stage;
    const posA = same.pos_a;
    const posB = same.pos_b;
    const head = block.att.heads[0];
    const predA = same.prediction_a;
    const predB = same.prediction_b;
    const token = escHtml(same.token_str);
    const outcome = same.logits_equal
      ? "logits_equal=true：两个位置的 logits 相同。"
      : "logits_equal=false：两个位置的 logits 不同；argmax 是否不同以列出的实际结果为准。";
    sameContent = "<div class=\\"formula-box\\">比较来源：固定独立输入 <code>" + escHtml(same.input_text) + "</code> (IDs: " + escHtml(JSON.stringify(same.input_ids)) + ")；不是自回归生成步骤。相同 Token " + token + " (ID=" + same.token_id + ") 出现在位置 " + posA + " 与 " + posB + "。</div>" +
      "<table class=\\"mat-table\\"><thead><tr><th>计算阶段</th><th>Pos " + posA + " (" + token + ")</th><th>Pos " + posB + " (" + token + ")</th><th>实测结果</th></tr></thead><tbody>" +
      "<tr><td class=\\"row-lbl\\">词嵌入</td><td>" + fmtVec(emb.tok_embeds[posA]) + "</td><td>" + fmtVec(emb.tok_embeds[posB]) + "</td><td>token_embeddings_equal=" + same.token_embeddings_equal + "</td></tr>" +
      "<tr><td class=\\"row-lbl\\">位置嵌入</td><td>" + fmtVec(emb.pos_embeds[posA]) + " (Pos=" + posA + ")</td><td>" + fmtVec(emb.pos_embeds[posB]) + " (Pos=" + posB + ")</td><td>position_embeddings_equal=" + same.position_embeddings_equal + "</td></tr>" +
      "<tr><td class=\\"row-lbl\\">融合输入 x_emb</td><td>" + fmtVec(emb.x_emb[posA]) + "</td><td>" + fmtVec(emb.x_emb[posB]) + "</td><td>词嵌入与位置嵌入相加</td></tr>" +
      "<tr><td class=\\"row-lbl\\">Head " + head.head_idx + " 注意力权重（首头示例）</td><td>" + fmtVec(head.attn_weights[posA]) + "</td><td>" + fmtVec(head.attn_weights[posB]) + "</td><td>每个位置可见的因果前缀长度不同</td></tr>" +
      "<tr><td class=\\"row-lbl\\">首个 Block 输出</td><td>" + fmtVec(block.res2.out[posA]) + "</td><td>" + fmtVec(block.res2.out[posB]) + "</td><td>取自独立比较 trace 的首层</td></tr>" +
      "<tr><td class=\\"row-lbl\\">Logits (" + CFG.vocab_size + " 维)</td><td>" + fmtVec(out.logits_all[posA]) + "</td><td>" + fmtVec(out.logits_all[posB]) + "</td><td>logits_equal=" + same.logits_equal + "；" + outcome + "</td></tr>" +
      "<tr><td class=\\"row-lbl\\">实际 Argmax</td><td>ID=" + predA.id + " ('" + escHtml(predA.token) + "')</td><td>ID=" + predB.id + " ('" + escHtml(predB.token) + "')</td><td>按比较记录展示，未假定两个预测不同</td></tr>" +
      "</tbody></table>";
  }

  return "<div class=\\"stage-card\\"><div class=\\"stage-header\\" onclick=\\"toggleStage(this)\\"><div class=\\"stage-title\\"><span>⚡ 跨步因果不变性与 K/V 比较</span><span class=\\"badge " + (verified ? "green" : "amber") + "\\">" + (verified ? "已按 tolerance 验证" : "未验证") + "</span></div></div>" +
    "<div class=\\"stage-body\\"><div class=\\"formula-box\\">" + invText + "<br/>" + kvExplanation + "</div><div class=\\"grid-2\\"><div class=\\"sub-box\\"><h4><span>逐张量最大绝对误差与检查结果</span></h4>" +
    "<table class=\\"mat-table\\"><thead><tr><th>比较张量</th><th>最大绝对误差</th><th>结果</th></tr></thead><tbody>" + (checkRows || "<tr><td colspan=\\"3\\">没有逐张量检查资料</td></tr>") + "</tbody></table></div>" + kvContent + "</div></div></div>" +
    "<div class=\\"stage-card\\"><div class=\\"stage-header\\" onclick=\\"toggleStage(this)\\"><div class=\\"stage-title\\"><span>🔍 同一 Token 位于不同位置的深度比较</span><span class=\\"badge amber\\">来源为固定独立输入</span></div></div><div class=\\"stage-body\\">" + sameContent + "</div></div>";
}

function renderWeightsView() {
  const w = TRACE.weights;

  const tokRows = w.tok_emb_weight.map((r, i) =>
    `<tr><td class="row-lbl">ID=${i} ('${escHtml(TRACE.vocab[i].token)}')</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  const posRows = w.pos_emb_weight.map((r, i) =>
    `<tr><td class="row-lbl">Pos=${i}</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  function renderMatrix(mat, isHeadSplit=true) {
    return mat.map((r, i) => {
      const lbl = isHeadSplit ? `OutDim ${i} (Head ${Math.floor(i/HEAD_DIM)}, d${i%HEAD_DIM})` : `OutDim ${i}`;
      return `<tr><td class="row-lbl">${lbl}</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`;
    }).join("");
  }

  const ohRows = w.out_head_weight.map((r, i) =>
    `<tr><td class="row-lbl">ID=${i} ('${escHtml(TRACE.vocab[i].token)}')</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  const dimHeaders = Array.from({length: EMB_DIM}, (_, i) => `<th>in${i}</th>`).join("");
  const outEmbParams = w.out_head_weight.length * EMB_DIM;
  const tokenEmbParams = w.tok_emb_weight.length * EMB_DIM;
  const positionEmbParams = w.pos_emb_weight.length * EMB_DIM;
  const blockCards = w.blocks.map((bw, blockIdx) => {
    const matrixHeaders = mat => Array.from({length: mat[0].length}, (_, i) => `<th>in${i}</th>`).join("");
    const queryBias = bw.W_query_bias || [];
    const keyBias = bw.W_key_bias || [];
    const valueBias = bw.W_value_bias || [];
    const matrixParams = mat => mat.length * (mat[0] || []).length;
    return `
      <div class="stage-card">
        <div class="stage-header" onclick="toggleStage(this)">
          <div class="stage-title"><span>📦 权重组 2：TransformerBlock ${blockIdx} 的 MultiHeadAttention 投影矩阵</span></div>
        </div>
        <div class="stage-body grid-4">
          <div class="sub-box">
            <h4><span>att.W_query.weight (${bw.W_query.length}×${bw.W_query[0].length})</span><span class="badge">${matrixParams(bw.W_query) + queryBias.length} params</span></h4>
            <table class="mat-table"><thead><tr><th>OutDim</th>${matrixHeaders(bw.W_query)}</tr></thead><tbody>${renderMatrix(bw.W_query)}</tbody></table>
            ${queryBias.length ? `<div class="formula-box">bias: ${fmtVec(queryBias)}</div>` : ""}
          </div>
          <div class="sub-box">
            <h4><span>att.W_key.weight (${bw.W_key.length}×${bw.W_key[0].length})</span><span class="badge">${matrixParams(bw.W_key) + keyBias.length} params</span></h4>
            <table class="mat-table"><thead><tr><th>OutDim</th>${matrixHeaders(bw.W_key)}</tr></thead><tbody>${renderMatrix(bw.W_key)}</tbody></table>
            ${keyBias.length ? `<div class="formula-box">bias: ${fmtVec(keyBias)}</div>` : ""}
          </div>
          <div class="sub-box">
            <h4><span>att.W_value.weight (${bw.W_value.length}×${bw.W_value[0].length})</span><span class="badge">${matrixParams(bw.W_value) + valueBias.length} params</span></h4>
            <table class="mat-table"><thead><tr><th>OutDim</th>${matrixHeaders(bw.W_value)}</tr></thead><tbody>${renderMatrix(bw.W_value)}</tbody></table>
            ${valueBias.length ? `<div class="formula-box">bias: ${fmtVec(valueBias)}</div>` : ""}
          </div>
          <div class="sub-box">
            <h4><span>att.out_proj.weight (${bw.out_proj_weight.length}×${bw.out_proj_weight[0].length})</span><span class="badge">${matrixParams(bw.out_proj_weight) + bw.out_proj_bias.length} params</span></h4>
            <table class="mat-table"><thead><tr><th>OutDim</th>${matrixHeaders(bw.out_proj_weight)}</tr></thead><tbody>${renderMatrix(bw.out_proj_weight, false)}</tbody></table>
            <div class="formula-box">out_proj.bias: ${fmtVec(bw.out_proj_bias)}</div>
          </div>
        </div>
      </div>`;
  }).join("");

  return `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title"><span>📦 权重组 1：词嵌入表 tok_emb.weight (${w.tok_emb_weight.length}×${EMB_DIM}) 与位置嵌入表 pos_emb.weight (${w.pos_emb_weight.length}×${EMB_DIM})</span></div>
      </div>
      <div class="stage-body grid-2">
        <div class="sub-box">
          <h4><span>tok_emb.weight (${w.tok_emb_weight.length} × ${EMB_DIM})</span><span class="badge">${tokenEmbParams} params</span></h4>
          <table class="mat-table">
            <thead><tr><th>词表项</th>${Array.from({length: EMB_DIM}, (_, i) => `<th>d${i}</th>`).join("")}</tr></thead>
            <tbody>${tokRows}</tbody>
          </table>
        </div>
        <div class="sub-box">
          <h4><span>pos_emb.weight (${w.pos_emb_weight.length} × ${EMB_DIM})</span><span class="badge">${positionEmbParams} params</span></h4>
          <table class="mat-table">
            <thead><tr><th>位置 Pos</th>${Array.from({length: EMB_DIM}, (_, i) => `<th>d${i}</th>`).join("")}</tr></thead>
            <tbody>${posRows}</tbody>
          </table>
        </div>
      </div>
    </div>

    ${blockCards}

    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title"><span>📦 权重组 3：输出投影头 out_head.weight (${w.out_head_weight.length}×${EMB_DIM})</span></div>
      </div>
      <div class="stage-body">
        <div class="sub-box" style="max-width:600px;">
          <h4><span>out_head.weight (${w.out_head_weight.length} × ${EMB_DIM}, bias=False)</span><span class="badge">${outEmbParams} params</span></h4>
          <table class="mat-table">
            <thead><tr><th>预测目标 Token</th>${dimHeaders}</tr></thead>
            <tbody>${ohRows}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

function switchTab(tabId) {
  const btns = document.querySelectorAll("#main-tabs .tab-btn");
  btns.forEach(b => b.classList.remove("active"));
  const stepMatch = /^step([0-9]+)$/.exec(tabId);
  const stepIndex = stepMatch ? Number(stepMatch[1]) : -1;
  const buttonIndex = stepIndex >= 0 ? stepIndex : (tabId === "compare" ? TRACE.steps.length : TRACE.steps.length + 1);
  if (btns[buttonIndex]) btns[buttonIndex].classList.add("active");

  const container = document.getElementById("view-container");
  if (stepIndex >= 0 && stepIndex < TRACE.steps.length) container.innerHTML = renderStepView(stepIndex);
  else if (tabId === "compare") container.innerHTML = renderCompareView();
  else if (tabId === "weights") container.innerHTML = renderWeightsView();
}

// 绑定悬浮验算提示框
document.addEventListener("mouseover", e => {
  const td = e.target.closest("td[data-tip]");
  const tipEl = document.getElementById("tooltip");
  if (td) {
    tipEl.textContent = td.getAttribute("data-tip").replace(/\\\\n/g, "\\n");
    tipEl.style.display = "block";
  } else {
    tipEl.style.display = "none";
  }
});
document.addEventListener("mousemove", e => {
  const tipEl = document.getElementById("tooltip");
  if (tipEl.style.display === "block") {
    tipEl.style.left = Math.min(window.innerWidth - 490, e.clientX + 16) + "px";
    tipEl.style.top = Math.min(window.innerHeight - 160, e.clientY + 16) + "px";
  }
});

renderTabButtons();
renderHeaderBadges();
switchTab("step0");
</script>
</body>
</html>
"""
    return html_template.replace("__TRACE_JSON__", json_str)


def export_dual_track_dissection(trace_data, output_dir):
    """把同一份快照分别渲染并写入 Markdown 报告与 HTML 检视器，返回两个路径。"""
    md_path = os.path.join(output_dir, "gpt2toy_dissection.md")
    html_path = os.path.join(output_dir, "gpt2toy_inspector.html")

    md_content = generate_markdown_report(trace_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    html_content = generate_html_inspector(trace_data)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return md_path, html_path
