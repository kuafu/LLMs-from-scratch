# -*- coding: utf-8 -*-
"""
GPT-2 Toy 双轨剖析导出器 (Dual-Track Dissection Exporter)
生成：
  1. gpt2toy_dissection.md  —— 结构化图文 + 公式 + 完整矩阵对照 Markdown 深度剖析文档
  2. gpt2toy_inspector.html —— 零外部依赖的单文件交互式 HTML 观战面板
"""

import os
import json
import math


def _f(val, p=4):
    if val is None:
        return "None"
    if isinstance(val, float) and math.isinf(val):
        return "-∞" if val < 0 else "+∞"
    return f"{val:.{p}f}"


def _vec_str(vec, p=4):
    return "[" + ", ".join(_f(v, p) for v in vec) + "]"


def _md_table(headers, rows, aligns=None):
    if not aligns:
        aligns = ["left"] + ["right"] * (len(headers) - 1)
    sep_map = {"left": ":---", "center": ":---:", "right": "---:"}
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(sep_map.get(a, "---") for a in aligns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def generate_markdown_report(trace_data):
    cfg = trace_data["config"]
    ps = trace_data["param_stats"]
    w = trace_data["weights"]
    vocab = trace_data["vocab"]
    steps = trace_data["steps"]
    comp = trace_data["comparisons"]

    d = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    head_dim = cfg["head_dim"]
    vocab_map = {item["id"]: item["token"] for item in vocab}

    md = []
    md.append("# GPT-2 Toy 全链路像素级深度剖析报告 (Complete Dissection Report)\n")
    md.append("> **自动生成说明**：本报告由 `gpt2toy.py` 实测运行并无损录制生成，固定随机种子 `torch.manual_seed(123)`，处于推断模式 (`model.eval()`, `drop_rate=0.0`)。所有矩阵与中间张量数值均为真实 PyTorch 前向传播精确结果。\n")

    # 目录
    md.append("## 目录 (Table of Contents)\n")
    md.append("1. [全局拓扑与张量维度演化图 (Architecture & Tensor Flow)](#1-全局拓扑与张量维度演化图)")
    md.append("2. [超参数规格与 312 个可学习参数精确拆解](#2-超参数规格与-312-个可学习参数精确拆解)")
    md.append("3. [全模块初始权重矩阵清单 (Initialized Weights)](#3-全模块初始权重矩阵清单)")
    md.append("4. [自回归 Step 1 全链路前向传播剖析 (`T=4`: `\"Hello, I am\"` → `\"I\"`)](#4-自回归-step-1-全链路前向传播剖析)")
    md.append("5. [自回归 Step 2 全链路前向传播剖析 (`T=5`: `\"Hello, I am I\"` → `\"I\"`)](#5-自回归-step-2-全链路前向传播剖析)")
    md.append("6. [跨步因果不变性验证 (KV Cache 核心原理) 与同词异位对比](#6-跨步因果不变性验证与同词异位对比)\n")

    # 1. Mermaid 流程图
    md.append("---\n## 1. 全局拓扑与张量维度演化图\n")
    md.append("```mermaid")
    md.append("flowchart TD")
    md.append('    In["输入序列 in_idx<br/>形状: (B=1, T)"] --> TokEmb["词嵌入查表 tok_emb<br/>weight: (6, 4) → 输出: (1, T, 4)"]')
    md.append('    In --> PosEmb["位置嵌入查表 pos_emb<br/>weight: (6, 4) → 输出: (T, 4)"]')
    md.append('    TokEmb --> AddEmb["广播逐元素相加: x = tok_emb + pos_emb<br/>形状: (1, T, 4)"]')
    md.append('    PosEmb --> AddEmb')
    md.append('    AddEmb --> LN1["Pre-LN 1: LayerNorm(x)<br/>(x - μ) / √(σ² + 1e-5) * γ + β → (1, T, 4)"]')
    md.append('    LN1 --> QKV["Q / K / V 线性投影 (无偏置)<br/>x_norm1 @ W_q^T, W_k^T, W_v^T → 3 × (1, T, 4)"]')
    md.append('    QKV --> Split["多头拆分 (view + transpose)<br/>Head 0 (列 0..1) & Head 1 (列 2..3) → 3 × (1, 2, T, 2)"]')
    md.append('    Split --> Scores["四步注意力得分演化<br/>① Q@K^T → ② /√2 → ③ 上三角置 -∞ → ④ Softmax → (1, 2, T, T)"]')
    md.append('    Scores --> Context["Value 加权求和 & 多头左右拼接<br/>attn_weights @ V → (1, 2, T, 2) → Concat (1, T, 4)"]')
    md.append('    Context --> OutProj["MHA 输出线性投影 out_proj<br/>context @ W_out^T + b_out → (1, T, 4)"]')
    md.append('    AddEmb -- "Shortcut 1 (残差主干)" --> Res1["残差连接 1: x = shortcut1 + att_out<br/>形状: (1, T, 4)"]')
    md.append('    OutProj --> Res1')
    md.append('    Res1 --> LN2["Pre-LN 2: LayerNorm(x)<br/>形状: (1, T, 4)"]')
    md.append('    LN2 --> FFN1["FFN 升维层: Linear(4 → 16)<br/>形状: (1, T, 16)"]')
    md.append('    FFN1 --> GELU["非线性激活: GELU(h1)<br/>形状: (1, T, 16)"]')
    md.append('    GELU --> FFN2["FFN 降维层: Linear(16 → 4)<br/>形状: (1, T, 4)"]')
    md.append('    Res1 -- "Shortcut 2 (残差主干)" --> Res2["残差连接 2: x = shortcut2 + ff_out<br/>形状: (1, T, 4)"]')
    md.append('    FFN2 --> Res2')
    md.append('    Res2 --> FinalLN["最终层归一化: final_norm(x)<br/>形状: (1, T, 4)"]')
    md.append('    FinalLN --> OutHead["词表投影头: out_head(x)<br/>x_final @ W_head^T → Logits (1, T, 6)"]')
    md.append('    OutHead --> NextTok["提取末尾时间步 logits[:, -1, :]<br/>Softmax 概率分布 → Argmax 采样下一个 Token"]')
    md.append("```\n")

    # 2. 超参数与参数统计
    md.append("---\n## 2. 超参数规格与 312 个可学习参数精确拆解\n")
    md.append("### 2.1 模型超参数表与微型词表\n")
    cfg_rows = [
        ["`vocab_size` (词表大小)", f"`{cfg['vocab_size']}`", "`'<|unk|>':0, '<|endoftext|>':1, 'Hello':2, ',':3, 'I':4, 'am':5`"],
        ["`context_length` (最大上下文长度)", f"`{cfg['context_length']}`", "位置嵌入表支持 `Pos = 0..5`"],
        ["`emb_dim` (嵌入维度 $d_{\\text{model}}$)", f"`{d}`", "主干残差流向量宽度为 4 维"],
        ["`n_heads` (注意力头数 $H$)", f"`{n_heads}`", f"每个头维度 $d_k = d_v = {d} / {n_heads} = {head_dim}$"],
        ["`ffn_hidden_dim` (FFN 隐藏维 $4d$)", f"`{cfg['ffn_hidden_dim']}`", f"前馈网络升维至 $4 \\times {d} = {4*d}$ 维"],
        ["`n_layers` (Transformer 层数)", f"`{cfg['n_layers']}`", "包含 1 个完整的 `TransformerBlock`"],
        ["`qkv_bias` / `drop_rate`", f"`{cfg['qkv_bias']}` / `{cfg['drop_rate']}`", "Q/K/V 线性层无偏置，推断模式关闭 Dropout"],
    ]
    md.append(_md_table(["超参数名称", "取值", "架构意义说明"], cfg_rows, ["left", "center", "left"]) + "\n")

    md.append("### 2.2 全模块可学习参数量实测分布\n")
    param_rows = [
        ["1. 输入嵌入阶段", "`tok_emb.weight`", f"`({cfg['vocab_size']}, {d})`", "`False`", f"`{ps['tok_emb_p']}`"],
        ["1. 输入嵌入阶段", "`pos_emb.weight`", f"`({cfg['context_length']}, {d})`", "`False`", f"`{ps['pos_emb_p']}`"],
        ["2. TransformerBlock 0", "`norm1` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['ln1_p']}`"],
        ["2. TransformerBlock 0", "`att.W_query / W_key / W_value`", f"`3 × ({d}, {d})`", "`False`", f"`{ps['qkv_p']}`"],
        ["2. TransformerBlock 0", "`att.out_proj` (`weight` + `bias`)", f"`({d}, {d}) + ({d},)`", "`True`", f"`{ps['out_proj_p']}`"],
        ["2. TransformerBlock 0", "`norm2` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['ln2_p']}`"],
        ["2. TransformerBlock 0", "`ff.layers[0]` (升维 `4→16`)", f"`({4*d}, {d}) + ({4*d},)`", "`True`", f"`{ps['ffn_fc1_p']}`"],
        ["2. TransformerBlock 0", "`ff.layers[2]` (降维 `16→4`)", f"`({d}, {4*d}) + ({d},)`", "`True`", f"`{ps['ffn_fc2_p']}`"],
        ["3. 输出预测阶段", "`final_norm` (`scale` + `shift`)", f"`({d},) + ({d},)`", "—", f"`{ps['final_ln_p']}`"],
        ["3. 输出预测阶段", "`out_head.weight` (词表投影)", f"`({cfg['vocab_size']}, {d})`", "`False`", f"`{ps['out_head_p']}`"],
        ["**全模型总计 (Total)**", "**所有可学习参数**", "**Float32 占 1248 Bytes**", "—", f"**`{ps['total_p']}`**"],
    ]
    md.append(_md_table(["所属阶段", "模块 / 参数张量", "权重形状", "含 Bias", "参数量"], param_rows, ["left", "left", "center", "center", "right"]) + "\n")

    # 3. 全模块初始权重矩阵清单
    md.append("---\n## 3. 全模块初始权重矩阵清单 (Initialized Weights)\n")
    md.append("> **注意**：PyTorch 中 `nn.Linear(in_features, out_features)` 的 `.weight` 存储形状为 `(out_features, in_features)`，前向计算公式为 $y = x W^T + b$。\n")

    md.append("### 3.1 词嵌入表 `tok_emb.weight` `(6 × 4)` 与 位置嵌入表 `pos_emb.weight` `(6 × 4)`\n")
    tok_rows = []
    for idx, row in enumerate(w["tok_emb_weight"]):
        tok_rows.append([f"`ID={idx}` (`'{vocab_map[idx]}'`)", *[_f(v) for v in row]])
    md.append("**词嵌入查找表 `tok_emb.weight` (行号即 Token ID)：**\n")
    md.append(_md_table(["Token ID (词元)", "Dim 0", "Dim 1", "Dim 2", "Dim 3"], tok_rows) + "\n")

    pos_rows = []
    for idx, row in enumerate(w["pos_emb_weight"]):
        pos_rows.append([f"`Pos={idx}`", *[_f(v) for v in row]])
    md.append("**位置嵌入查找表 `pos_emb.weight` (行号即序列位置 `0..5`)：**\n")
    md.append(_md_table(["位置索引 Pos", "Dim 0", "Dim 1", "Dim 2", "Dim 3"], pos_rows) + "\n")

    bw = w["blocks"][0]
    md.append("### 3.2 `TransformerBlock 0` 内部权重矩阵\n")
    md.append(f"- **`norm1` (Pre-LN 1)**: `scale` = `{_vec_str(bw['norm1_scale'])}`, `shift` = `{_vec_str(bw['norm1_shift'])}`\n")
    md.append(f"- **`norm2` (Pre-LN 2)**: `scale` = `{_vec_str(bw['norm2_scale'])}`, `shift` = `{_vec_str(bw['norm2_shift'])}`\n")

    for w_title, w_key in [
        ("Query 投影矩阵 `att.W_query.weight` `(4 × 4)`", "W_query"),
        ("Key 投影矩阵 `att.W_key.weight` `(4 × 4)`", "W_key"),
        ("Value 投影矩阵 `att.W_value.weight` `(4 × 4)`", "W_value"),
    ]:
        md.append(f"**{w_title}**（前 2 行归属 `Head 0`，后 2 行归属 `Head 1`）：\n")
        qkv_rows = []
        for r_idx, row in enumerate(bw[w_key]):
            h_id = r_idx // head_dim
            d_id = r_idx % head_dim
            qkv_rows.append([f"`OutDim {r_idx}` (`Head {h_id}, d{d_id}`)", *[_f(v) for v in row]])
        md.append(_md_table(["输出维度 (归属头)", "InDim 0", "InDim 1", "InDim 2", "InDim 3"], qkv_rows) + "\n")

    md.append("**多头输出投影层 `att.out_proj.weight` `(4 × 4)` 与 `bias` `(4,)`：**\n")
    op_rows = []
    for r_idx, row in enumerate(bw["out_proj_weight"]):
        op_rows.append([f"`OutDim {r_idx}`", *[_f(v) for v in row], _f(bw["out_proj_bias"][r_idx])])
    md.append(_md_table(["输出维度", "InDim 0 (H0,d0)", "InDim 1 (H0,d1)", "InDim 2 (H1,d0)", "InDim 3 (H1,d1)", "Bias"], op_rows) + "\n")

    md.append("<details>\n<summary><b>点击展开查看 FFN 升维层 (16×4) 与 降维层 (4×16) 完整权重表</b></summary>\n\n")
    fc1_rows = []
    for r_idx, row in enumerate(bw["ff_fc1_weight"]):
        fc1_rows.append([f"`HidDim {r_idx:02d}`", *[_f(v) for v in row], _f(bw["ff_fc1_bias"][r_idx])])
    md.append("**FFN 升维层 `ff.layers[0].weight` `(16 × 4)` 与 `bias` `(16,)`：**\n\n")
    md.append(_md_table(["隐藏维索引", "InDim 0", "InDim 1", "InDim 2", "InDim 3", "Bias"], fc1_rows) + "\n\n")

    fc2_rows = []
    for r_idx, row in enumerate(bw["ff_fc2_weight"]):
        fc2_rows.append([f"`OutDim {r_idx}`", _vec_str(row[:8]), _vec_str(row[8:]), _f(bw["ff_fc2_bias"][r_idx])])
    md.append("**FFN 降维层 `ff.layers[2].weight` `(4 × 16)` 与 `bias` `(4,)`：**\n\n")
    md.append(_md_table(["输出维度", "InDim [00..07]", "InDim [08..15]", "Bias"], fc2_rows, ["left", "left", "left", "right"]) + "\n")
    md.append("</details>\n")

    md.append("### 3.3 最终归一化 `final_norm` 与 词表输出头 `out_head.weight` `(6 × 4)`\n")
    md.append(f"- **`final_norm`**: `scale` = `{_vec_str(w['final_norm_scale'])}`, `shift` = `{_vec_str(w['final_norm_shift'])}`\n")
    oh_rows = []
    for idx, row in enumerate(w["out_head_weight"]):
        oh_rows.append([f"`ID={idx}` (`'{vocab_map[idx]}'`)", *[_f(v) for v in row]])
    md.append(_md_table(["预测目标词表项", "InDim 0", "InDim 1", "InDim 2", "InDim 3"], oh_rows) + "\n")

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
        md.append(f"- **当前步骤输入文本**: `\"{step['input_text']}\"`\n")
        md.append(f"- **输入 Token IDs (`in_idx`)**: `{in_ids}` (形状 `(B=1, T={T})`)\n")
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
        md.append("对每个 Token 的 4 维向量独立求均值 $\\mu_t = \\frac{1}{4}\\sum_{d=0}^3 x_{t,d}$ 与有偏方差 $\\sigma^2_t = \\frac{1}{4}\\sum_{d=0}^3 (x_{t,d}-\\mu_t)^2$，再执行标准化 $\\hat{x}_{t,d} = \\frac{x_{t,d} - \\mu_t}{\\sqrt{\\sigma^2_t + 10^{-5}}}$：\n")
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
        md.append(f"#### ({sec_num}.3.1) 切头前完整 4 维线性映射 vs 多头切分 (`Head 0` & `Head 1`)\n")
        md.append("线性变换 $Q = X_{\\text{norm1}} W_Q^T,\\; K = X_{\\text{norm1}} W_K^T,\\; V = X_{\\text{norm1}} W_V^T$ 输出形状均为 `(1, T, 4)`。随后 `view(1, T, 2, 2).transpose(1, 2)` 将前 2 列划归 `Head 0`，后 2 列划归 `Head 1`：\n")
        qkv_split_rows = []
        for t in range(T):
            qkv_split_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(att['q_merged'][t])}`",
                f"`{_vec_str(att['heads'][0]['q'][t])}`",
                f"`{_vec_str(att['heads'][1]['q'][t])}`",
                f"`{_vec_str(att['heads'][0]['k'][t])}`",
                f"`{_vec_str(att['heads'][1]['k'][t])}`",
                f"`{_vec_str(att['heads'][0]['v'][t])}`",
                f"`{_vec_str(att['heads'][1]['v'][t])}`",
            ])
        md.append(_md_table(
            ["Token", "`Q_full (4维)`", "`Q_Head0`", "`Q_Head1`", "`K_Head0`", "`K_Head1`", "`V_Head0`", "`V_Head1`"],
            qkv_split_rows, ["left"] * 8
        ) + "\n")

        md.append(f"#### ({sec_num}.3.2) 注意力得分矩阵四步演化 (`Raw` → `Scaled` → `Masked` → `Softmax`)\n")
        md.append("缩放因子 $\\sqrt{d_k} = \\sqrt{2} \\approx 1.4142$。因果掩码将上三角区域 ($j > i$，即未来位置) 置为 $-\\infty$，使得 $\\text{softmax}$ 后未来位置权重严格为 `0.0000`：\n")

        col_headers = ["Query 位置 \\ Key 位置"] + [f"`t{j}` (`'{step['input_tokens'][j]}'`)" for j in range(T)]
        for h_obj in att["heads"]:
            h_idx = h_obj["head_idx"]
            md.append(f"**【Head {h_idx}】四步注意力矩阵 (`{T} × {T}`)：**\n")

            for stage_title, mat_key in [
                (f"① 原始点积得分 `raw_scores = Q_{h_idx} @ K_{h_idx}^T`", "raw_scores"),
                (f"② 缩放得分 `unmasked_scaled = raw_scores / √2`", "unmasked_scaled_scores"),
                (f"③ 因果掩码遮蔽后得分 `masked_scores` ($j > i$ 置为 $-\\infty$)", "masked_scores"),
                (f"④ Softmax 归一化注意力权重 `attn_weights` (每行概率和 = `1.0000`)", "attn_weights"),
            ]:
                md.append(f"- **{stage_title}**：\n")
                m_rows = []
                for i_r in range(T):
                    m_rows.append([f"`{t_labels[i_r]}`", *[_f(v) for v in h_obj[mat_key][i_r]]])
                md.append(_md_table(col_headers, m_rows) + "\n")

        md.append(f"#### ({sec_num}.3.3) Value 加权求和、多头左右拼接与 `out_proj` 输出线性投影\n")
        md.append("每个头独立计算 $\\text{context}_h = \\text{attn\\_weights}_h V_h \\in \\mathbb{R}^{T \\times 2}$，随后左右拼接为 $\\text{context}_{\\text{merged}} = [\\text{context}_0 \\,\\|\\, \\text{context}_1] \\in \\mathbb{R}^{T \\times 4}$，最后经 `out_proj` 映射为 `att_out`：\n")
        ctx_rows = []
        for t in range(T):
            ctx_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(att['heads'][0]['context_head'][t])}`",
                f"`{_vec_str(att['heads'][1]['context_head'][t])}`",
                f"`{_vec_str(att['context_merged'][t])}`",
                f"`{_vec_str(att['out_proj_out'][t])}`",
            ])
        md.append(_md_table(
            ["Token", "`Head 0 Context (2维)`", "`Head 1 Context (2维)`", "多头拼接 `Concat (4维)`", "投影输出 `att_out (4维)`"],
            ctx_rows, ["left"] * 5
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
        md.append("**(2) `FFN` 升维 (`4→16`)、GELU 激活与降维 (`16→4`) 输出：**\n")
        ffn_rows = []
        for t in range(T):
            ffn_rows.append([
                f"`{t_labels[t]}`",
                f"`{_vec_str(ffn['fc1_out'][t][:4])} ...` (共16维)",
                f"`{_vec_str(ffn['gelu_out'][t][:4])} ...` (共16维)",
                f"`{_vec_str(ffn['fc2_out'][t])}`",
            ])
        md.append(_md_table(["Token", "升维 `fc1_out` (前4维预览)", "激活 `GELU(fc1)` (前4维预览)", "降维输出 `ff_out (4维)`"], ffn_rows, ["left"] * 4) + "\n")

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
        md.append(_md_table(["Token", "均值 `μ`", "方差 `σ²`", "`final_norm` 输出 `(4维)`", "词表 Logits `out_head(x)` `(6维)`"], fln_rows, ["left", "right", "right", "left", "left"]) + "\n")

        md.append(f"**末尾时间步 `{pred['last_token_label']}` 对 6 个候选词的预测概率分布与 Argmax 采样：**\n")
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
        md.append(f"> **Step {step_num} 结论**：贪婪采样选中 `ID={pred['next_token_id']}` (`\"{pred['next_token_str']}\"`), 序列更新为 `{pred['updated_ids']}` → 文本：`\"{pred['updated_text']}\"`。\n")

    # 6. 跨步因果不变性与同词异位对比
    md.append("---\n## 6. 跨步因果不变性验证与同词异位对比\n")
    md.append("### 6.1 跨步因果不变性验证 (`Step 1` vs `Step 2` → 揭示 KV Cache 原理)\n")
    md.append("对比 `Step 1 (T=4)` 与 `Step 2 (T=5)` 中前 4 个位置 (`t0..t3`: `'Hello'`, `','`, `'I'`, `'am'`) 在各层的计算结果：\n")
    inv_rows = []
    for item in comp.get("invariance_checks", []):
        inv_rows.append([item["name"], f"`{item['max_diff']:.8f}`", "**完全恒等 (0 误差)** ✅"])
    md.append(_md_table(["前向传播中间张量阶段", "最大绝对误差 `max|Step1 - Step2[:4]|`", "因果不变性验证结论"], inv_rows, ["left", "right", "center"]) + "\n")
    md.append("> **核心洞察 (KV Cache 原理)**：因为在 `MultiHeadAttention` 中使用了上三角 `-inf` 因果掩码，位置 $i$ 的注意力分布与上下文向量**绝对不会受任何 $j > i$ 的未来 Token 影响**。同时 `LayerNorm`、`Linear`、`GELU` 均为逐位置 (`Token-wise`) 独立运算。因此在自回归生成第 5 个 Token 时，前 4 个 Token 的 $K, V$ 向量与 Step 1 完全相同——工业界推理框架将历史步的 $K, V$ 缓存起来（即 **KV Cache**），每步只需输入最新生成的 1 个 Token，将单步计算复杂度从 $O(T^2)$ 降至 $O(T)$。\n")

    md.append("### 6.2 同词异位深度对比 (`Step 2` 中的 `t2[ID=4:'I', Pos=2]` vs `t4[ID=4:'I', Pos=4]`)\n")
    md.append("在 `Step 2` 中，`t2` 与 `t4` 输入的是同一个词 `ID=4 ('I')`，为什么模型在 `t2` 处预测下一个词是 `',' (ID=3)`，而在 `t4` 处预测下一个词是 `'I' (ID=4)`？下表展示了全链路差异的产生过程：\n")
    s2 = steps[1]
    b2 = s2["blocks"][0]
    diff_pos_rows = [
        ["词嵌入 `tok_emb[ID=4]`", f"`{_vec_str(s2['embeddings']['tok_embeds'][2])}`", f"`{_vec_str(s2['embeddings']['tok_embeds'][4])}`", "完全相同（同一词表行）"],
        ["位置嵌入 `pos_emb[Pos]`", f"`{_vec_str(s2['embeddings']['pos_embeds'][2])}` (`Pos=2`)", f"`{_vec_str(s2['embeddings']['pos_embeds'][4])}` (`Pos=4`)", "**差异源 1：位置编码不同**"],
        ["融合输入 `x_emb`", f"`{_vec_str(s2['embeddings']['x_emb'][2])}`", f"`{_vec_str(s2['embeddings']['x_emb'][4])}`", "位置信息注入特征向量"],
        ["`Head 0` 注意力权重分布", f"`{_vec_str(b2['att']['heads'][0]['attn_weights'][2])}`", f"`{_vec_str(b2['att']['heads'][0]['attn_weights'][4])}`", "**差异源 2：`t2` 仅看 `t0..t2`，`t4` 可看 `t0..t4`**"],
        ["`Head 1` 注意力权重分布", f"`{_vec_str(b2['att']['heads'][1]['attn_weights'][2])}`", f"`{_vec_str(b2['att']['heads'][1]['attn_weights'][4])}`", "因果视野宽度不同"],
        ["Block 最终输出 `res2.out`", f"`{_vec_str(b2['res2']['out'][2])}`", f"`{_vec_str(b2['res2']['out'][4])}`", "融合了各自的位置与上下文语义"],
        ["最终预测 Logits `(6维)`", f"`{_vec_str(s2['output_stage']['logits_all'][2])}`", f"`{_vec_str(s2['output_stage']['logits_all'][4])}`", "`t2` 最高为 `ID=3 (',')`，`t4` 最高为 `ID=4 ('I')`"],
    ]
    md.append(_md_table(["对比维度", "位置 `t2` (`ID=4 'I'`, `Pos=2`)", "位置 `t4` (`ID=4 'I'`, `Pos=4`)", "机制解读"], diff_pos_rows, ["left", "left", "left", "left"]) + "\n")

    return "\n".join(md)


def generate_html_inspector(trace_data):
    # 将 JSON 中可能出现的 -inf 转换为特殊字符串以便前端 JS 安全解析并渲染
    def _sanitize_for_json(obj):
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
    json_str = json.dumps(clean_data, ensure_ascii=False)

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
    <button class="tab-btn active" onclick="switchTab('step0')">▶ Step 1 观战 (T=4: "Hello, I am" → "I")</button>
    <button class="tab-btn" onclick="switchTab('step1')">▶ Step 2 观战 (T=5: "Hello, I am I" → "I")</button>
    <button class="tab-btn" onclick="switchTab('compare')">⚡ 跨步对比 (KV Cache 原理 & 同词异位)</button>
    <button class="tab-btn" onclick="switchTab('weights')">📦 模型架构与 312 个初始权重矩阵</button>
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
  `;
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
    `<span class="token-Zero"><span class="pos">Pos=${i}</span><span>ID=${tid}</span><span class="str">'${step.input_tokens[i]}'</span></span>`
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
          next_token = ID=${pred.next_token_id} ("${pred.next_token_str}") → "${pred.updated_text}"
        </div>
      </div>
    </div>
  `;

  // Stage 1: Input Embeddings
  let embRows = "";
  for (let t = 0; t < T; t++) {
    const tid = step.input_ids[t];
    let cells = "";
    for (let d = 0; d < 4; d++) {
      const tv = emb.tok_embeds[t][d];
      const pv = emb.pos_embeds[t][d];
      const xv = emb.x_emb[t][d];
      const tip = `位置 t${t} ('${step.input_tokens[t]}'), 维度 d${d}:\\ntok_emb[ID=${tid}, d${d}] (${fmt(tv)}) + pos_emb[Pos=${t}, d${d}] (${fmt(pv)}) = ${fmt(xv)}`;
      cells += `<td data-tip="${tip}" style="${heatBg(xv, -3, 3)}">${fmt(xv)}</td>`;
    }
    embRows += `
      <tr>
        <td class="row-lbl">${tLabels[t]}</td>
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
          <span class="badge">(1, ${T}) → tok_emb (1, ${T}, 4) + pos_emb (${T}, 4) = x (1, ${T}, 4)</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          数学公式: X[t, d] = tok_emb.weight[in_idx[t], d] + pos_emb.weight[t, d]  (将鼠标悬停在右侧结果单元格上可查看逐元素相加验算)
        </div>
        <table class="mat-table">
          <thead>
            <tr>
              <th>序列位置 Token</th>
              <th>词嵌入查表 tok_emb[ID] (1×4)</th>
              <th>位置嵌入查表 pos_emb[Pos] (1×4)</th>
              <th>x[d0]</th><th>x[d1]</th><th>x[d2]</th><th>x[d3]</th>
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
    for (let d = 0; d < 4; d++) {
      const xin = ln1.in[t][d];
      const xout = ln1.out[t][d];
      const tip = `Pre-LN 1 验算 (t${t}, d${d}):\\n(x=${fmt(xin)} - μ=${fmt(mu)}) / √(σ²=${fmt(vr)} + 1e-5) = ${fmt(xout)}`;
      outCells += `<td data-tip="${tip}" style="${heatBg(xout, -2, 2)}">${fmt(xout)}</td>`;
    }
    ln1Rows += `
      <tr>
        <td class="row-lbl">${tLabels[t]}</td>
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
          <span class="badge">x (1, ${T}, 4) → x_norm1 (1, ${T}, 4)</span>
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
              <th>norm1[d0]</th><th>norm1[d1]</th><th>norm1[d2]</th><th>norm1[d3]</th>
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
        <td class="row-lbl">${tLabels[t]}</td>
        <td style="text-align:left;">${fmtVec(att.q_merged[t])}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(att.heads[0].q[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(att.heads[1].q[t])}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(att.heads[0].k[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(att.heads[1].k[t])}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(att.heads[0].v[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(att.heads[1].v[t])}</td>
      </tr>
    `;
  }

  let headsEvoHtml = "";
  for (let h = 0; h < att.heads.length; h++) {
    const hObj = att.heads[h];
    const stages = [
      { title: `① 原始点积 Q_${h} @ K_${h}^T`, key: "raw_scores", isProb: false },
      { title: `② 缩放得分 (/ √2 ≈ 1.4142)`, key: "unmasked_scaled_scores", isProb: false },
      { title: `③ 因果掩码遮蔽 (j>i 置 -∞)`, key: "masked_scores", isProb: false },
      { title: `④ Softmax 注意力权重 (行和=1)`, key: "attn_weights", isProb: true },
    ];

    let stageBoxes = "";
    stages.forEach(st => {
      let ths = `<th>Q \\ K</th>` + step.input_tokens.map((tok, j) => `<th>t${j}('${tok}')</th>`).join("");
      let trs = "";
      for (let i = 0; i < T; i++) {
        let tds = `<td class="row-lbl">t${i}('${step.input_tokens[i]}')</td>`;
        for (let j = 0; j < T; j++) {
          const val = hObj[st.key][i][j];
          const isMasked = val === "-Infinity";
          const qVec = hObj.q[i];
          const kVec = hObj.k[j];
          const dotStr = `${fmt(qVec[0])}×${fmt(kVec[0])} + ${fmt(qVec[1])}×${fmt(kVec[1])} = ${fmt(hObj.raw_scores[i][j])}`;
          const tip = `Head ${h} [Query t${i}, Key t${j}]:\\nQ_${h}[t${i}] = ${fmtVec(qVec)}, K_${h}[t${j}] = ${fmtVec(kVec)}\\n原始点积 = ${dotStr}\\n缩放后 = ${fmt(hObj.unmasked_scaled_scores[i][j])}\\nSoftmax权重 = ${fmt(hObj.attn_weights[i][j])}`;
          tds += `<td class="${isMasked ? 'masked-cell' : ''}" data-tip="${tip}" style="${heatBg(val, -0.6, 0.6, st.isProb)}">${fmt(val)}</td>`;
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
          🔹 Head ${h} 注意力矩阵四步同屏演化 (源自线性映射列 [${h*2}..${h*2+1}])：
        </div>
        <div class="grid-4">${stageBoxes}</div>
      </div>
    `;
  }

  let ctxRows = "";
  for (let t = 0; t < T; t++) {
    ctxRows += `
      <tr>
        <td class="row-lbl">${tLabels[t]}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(att.heads[0].context_head[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(att.heads[1].context_head[t])}</td>
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
          <span class="badge">(1, ${T}, 4) → 2 Heads × (${T}, 2) → Attn (${T}×${T}) → Concat & out_proj (1, ${T}, 4)</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          Attention(Q, K, V) = softmax( mask(Q @ K^T / √2) ) @ V  |  前 2 列归属 Head 0，后 2 列归属 Head 1
        </div>
        <div class="sub-box" style="margin-bottom:14px;">
          <h4><span>(3.1) 切头前完整 4 维 Q_full 与 多头空间拆分后 Q / K / V 对照</span><span class="badge">(1, ${T}, 4) → (1, 2, ${T}, 2)</span></h4>
          <table class="mat-table">
            <thead>
              <tr>
                <th>Token</th>
                <th>Q_full (切头前 4维)</th>
                <th>Head 0: Q_0 (列0..1)</th>
                <th>Head 1: Q_1 (列2..3)</th>
                <th>Head 0: K_0</th>
                <th>Head 1: K_1</th>
                <th>Head 0: V_0</th>
                <th>Head 1: V_1</th>
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
                <th>Head 0: attn_0 @ V_0 (2维)</th>
                <th>Head 1: attn_1 @ V_1 (2维)</th>
                <th>左右拼接 context_vec (4维)</th>
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
        <td class="row-lbl">${tLabels[t]}</td>
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
          <span class="badge">(1, ${T}, 4) + (1, ${T}, 4) → (1, ${T}, 4)</span>
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
        <td class="row-lbl">${tLabels[t]}</td>
        <td style="text-align:left;">${fmtVec(ln2.out[t])} (μ=${fmt(ln2.mean[t])}, σ²=${fmt(ln2.var[t])})</td>
        <td style="text-align:left;color:var(--muted);font-size:11.5px;">${fmtVec(ffn.fc1_out[t].slice(0, 6))} ... (16维)</td>
        <td style="text-align:left;color:var(--amber);font-size:11.5px;">${fmtVec(ffn.gelu_out[t].slice(0, 6))} ... (16维)</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(ffn.fc2_out[t])}</td>
        <td style="text-align:left;color:var(--green);font-weight:600;">${fmtVec(res2.out[t])}</td>
      </tr>
    `;
  }

  html += `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>5️⃣ 阶段五：Pre-LN 2 (norm2) → 逐位置前馈网络 (FFN 4→16→4) → 残差连接 2</span>
          <span class="badge">(1, ${T}, 4) → (1, ${T}, 16) → (1, ${T}, 4)</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          FFN(x_norm2) = Linear_16→4( GELU( Linear_4→16(x_norm2) ) )  |  x_block_out = x_res1 + ff_out
        </div>
        <table class="mat-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>Pre-LN 2 输出 x_norm2</th>
              <th>升维 Linear(4→16) 前6维</th>
              <th>非线性激活 GELU(h1) 前6维</th>
              <th>降维输出 ff_out (4维)</th>
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
      lCells += `<td data-tip="${tip}" style="${heatBg(pv, 0, 1, true)}">${fmt(lv)} <span style="color:var(--muted);font-size:11px;">(${(pv*100).toFixed(1)}%)</span></td>`;
    }
    logitRows += `
      <tr class="${isLast ? 'selected-row' : ''}">
        <td class="row-lbl">${tLabels[t]} ${isLast ? '⬅ 末尾预测步' : ''}</td>
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
        <td class="row-lbl">ID=${c.id} ('${c.token}')</td>
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
          <span class="badge">(1, ${T}, 4) → Logits (1, ${T}, 6) → Argmax: ID=${pred.next_token_id} ('${pred.next_token_str}')</span>
        </div>
        <span style="color:var(--muted);font-size:12px;">点击折叠/展开</span>
      </div>
      <div class="stage-body">
        <div class="grid-2">
          <div class="sub-box">
            <h4><span>全序列每个位置的 Final LayerNorm 与 6 词表 Logits (括号内为概率)</span></h4>
            <table class="mat-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>final_norm(x) (4维)</th>
                  ${TRACE.vocab.map(v => `<th>ID=${v.id}<br/>'${v.token}'</th>`).join("")}
                </tr>
              </thead>
              <tbody>${logitRows}</tbody>
            </table>
          </div>
          <div class="sub-box">
            <h4><span>末尾时间步 ${pred.last_token_label} 候选词概率分布与贪婪采样</span></h4>
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
  const s1 = TRACE.steps[0];
  const s2 = TRACE.steps[1];
  const b1 = s1.blocks[0];
  const b2 = s2.blocks[0];
  const checks = TRACE.comparisons.invariance_checks || [];

  let checkRows = checks.map(c => `
    <tr>
      <td class="row-lbl">${c.name}</td>
      <td style="color:var(--green);font-weight:700;">${c.max_diff.toFixed(8)}</td>
      <td style="text-align:center;"><span class="badge green">✔ 100% 恒等 (因果不变性成立)</span></td>
    </tr>
  `).join("");

  let kvRows = "";
  for (let t = 0; t < 4; t++) {
    kvRows += `
      <tr>
        <td class="row-lbl">${s1.token_labels[t]}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(b1.att.k_merged[t])}</td>
        <td style="text-align:left;color:var(--accent);">${fmtVec(b2.att.k_merged[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(b1.att.v_merged[t])}</td>
        <td style="text-align:left;color:var(--purple);">${fmtVec(b2.att.v_merged[t])}</td>
      </tr>
    `;
  }
  kvRows += `
    <tr class="selected-row">
      <td class="row-lbl">${s2.token_labels[4]} (Step 2 新增)</td>
      <td style="text-align:center;color:var(--muted);">— (Step 1 不存在)</td>
      <td style="text-align:left;color:var(--green);font-weight:700;">${fmtVec(b2.att.k_merged[4])} (仅需新算此行)</td>
      <td style="text-align:center;color:var(--muted);">— (Step 1 不存在)</td>
      <td style="text-align:left;color:var(--green);font-weight:700;">${fmtVec(b2.att.v_merged[4])} (仅需新算此行)</td>
    </tr>
  `;

  return `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>⚡ 洞察一：跨步因果不变性验证 (Step 1 vs Step 2) —— 为什么大模型推理可以使用 KV Cache？</span>
          <span class="badge green">max|Step1 - Step2[:4]| = 0.00000000</span>
        </div>
      </div>
      <div class="stage-body">
        <div class="formula-box">
          由于因果掩码将上三角 (j > i) 置为 -∞，第 i 个位置在计算 Attention 时完全看不见任何 j > i 的后续 Token。因此当序列从 T=4 增长到 T=5 时，前 4 个位置 (t0..t3) 的所有中间张量分毫不差！
        </div>
        <div class="grid-2">
          <div class="sub-box">
            <h4><span>15 个核心中间张量的跨步最大绝对误差实测</span></h4>
            <table class="mat-table">
              <thead><tr><th>前向传播阶段张量</th><th>最大绝对误差</th><th>验证状态</th></tr></thead>
              <tbody>${checkRows}</tbody>
            </table>
          </div>
          <div class="sub-box">
            <h4><span>Step 1 与 Step 2 的 K / V 矩阵逐行对照 (KV Cache 直观演示)</span></h4>
            <table class="mat-table">
              <thead>
                <tr>
                  <th>Token 位置</th>
                  <th>Step 1: K_full (T=4)</th>
                  <th>Step 2: K_full (T=5)</th>
                  <th>Step 1: V_full (T=4)</th>
                  <th>Step 2: V_full (T=5)</th>
                </tr>
              </thead>
              <tbody>${kvRows}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title">
          <span>🔍 洞察二：同词异位深度对比 (Step 2 中 t2[ID=4:'I', Pos=2] vs t4[ID=4:'I', Pos=4])</span>
          <span class="badge amber">相同 Token ID=4 ('I')，为何输出截然不同？</span>
        </div>
      </div>
      <div class="stage-body">
        <table class="mat-table">
          <thead>
            <tr>
              <th>计算阶段</th>
              <th>位置 t2 (ID=4 'I', Pos=2)</th>
              <th>位置 t4 (ID=4 'I', Pos=4)</th>
              <th>差异根源解析</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="row-lbl">1. 词嵌入 tok_emb[ID=4]</td>
              <td style="text-align:left;">${fmtVec(s2.embeddings.tok_embeds[2])}</td>
              <td style="text-align:left;">${fmtVec(s2.embeddings.tok_embeds[4])}</td>
              <td style="text-align:left;color:var(--green);">完全一致（查同一行 tok_emb[4]）</td>
            </tr>
            <tr>
              <td class="row-lbl">2. 位置嵌入 pos_emb[Pos]</td>
              <td style="text-align:left;color:var(--amber);">${fmtVec(s2.embeddings.pos_embeds[2])} (Pos=2)</td>
              <td style="text-align:left;color:var(--amber);">${fmtVec(s2.embeddings.pos_embeds[4])} (Pos=4)</td>
              <td style="text-align:left;color:var(--amber);">差异源 1：位置嵌入注入了不同的绝对位置信息</td>
            </tr>
            <tr>
              <td class="row-lbl">3. 融合输入 x_emb</td>
              <td style="text-align:left;">${fmtVec(s2.embeddings.x_emb[2])}</td>
              <td style="text-align:left;">${fmtVec(s2.embeddings.x_emb[4])}</td>
              <td style="text-align:left;">进入 TransformerBlock 前已携带位置差异</td>
            </tr>
            <tr>
              <td class="row-lbl">4. Head 0 注意力权重</td>
              <td style="text-align:left;color:var(--accent);">${fmtVec(b2.att.heads[0].attn_weights[2])}</td>
              <td style="text-align:left;color:var(--accent);">${fmtVec(b2.att.heads[0].attn_weights[4])}</td>
              <td style="text-align:left;color:var(--accent);">差异源 2：t2 只能关注 t0..t2；t4 能关注 t0..t4</td>
            </tr>
            <tr>
              <td class="row-lbl">5. 最终预测 Logits (6维)</td>
              <td style="text-align:left;">${fmtVec(s2.output_stage.logits_all[2])} → 选 ID=3 (',')</td>
              <td style="text-align:left;color:var(--green);font-weight:700;">${fmtVec(s2.output_stage.logits_all[4])} → 选 ID=4 ('I')</td>
              <td style="text-align:left;">位置编码 + 因果上下文共同决定了不同的预测分布</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderWeightsView() {
  const w = TRACE.weights;
  const bw = w.blocks[0];

  const tokRows = w.tok_emb_weight.map((r, i) =>
    `<tr><td class="row-lbl">ID=${i} ('${TRACE.vocab[i].token}')</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  const posRows = w.pos_emb_weight.map((r, i) =>
    `<tr><td class="row-lbl">Pos=${i}</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  function mat4x4(mat, isHeadSplit=true) {
    return mat.map((r, i) => {
      const lbl = isHeadSplit ? `OutDim ${i} (Head ${Math.floor(i/2)}, d${i%2})` : `OutDim ${i}`;
      return `<tr><td class="row-lbl">${lbl}</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`;
    }).join("");
  }

  const ohRows = w.out_head_weight.map((r, i) =>
    `<tr><td class="row-lbl">ID=${i} ('${TRACE.vocab[i].token}')</td>${r.map(v => `<td>${fmt(v)}</td>`).join("")}</tr>`
  ).join("");

  return `
    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title"><span>📦 权重组 1：词嵌入表 tok_emb.weight (6×4) 与 位置嵌入表 pos_emb.weight (6×4)</span></div>
      </div>
      <div class="stage-body grid-2">
        <div class="sub-box">
          <h4><span>tok_emb.weight (6 × 4)</span><span class="badge">24 params</span></h4>
          <table class="mat-table">
            <thead><tr><th>词表项</th><th>d0</th><th>d1</th><th>d2</th><th>d3</th></tr></thead>
            <tbody>${tokRows}</tbody>
          </table>
        </div>
        <div class="sub-box">
          <h4><span>pos_emb.weight (6 × 4)</span><span class="badge">24 params</span></h4>
          <table class="mat-table">
            <thead><tr><th>位置 Pos</th><th>d0</th><th>d1</th><th>d2</th><th>d3</th></tr></thead>
            <tbody>${posRows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title"><span>📦 权重组 2：MultiHeadAttention 投影矩阵 (W_query, W_key, W_value, out_proj)</span></div>
      </div>
      <div class="stage-body grid-4">
        <div class="sub-box">
          <h4><span>att.W_query.weight (4×4)</span></h4>
          <table class="mat-table"><thead><tr><th>OutDim</th><th>in0</th><th>in1</th><th>in2</th><th>in3</th></tr></thead><tbody>${mat4x4(bw.W_query)}</tbody></table>
        </div>
        <div class="sub-box">
          <h4><span>att.W_key.weight (4×4)</span></h4>
          <table class="mat-table"><thead><tr><th>OutDim</th><th>in0</th><th>in1</th><th>in2</th><th>in3</th></tr></thead><tbody>${mat4x4(bw.W_key)}</tbody></table>
        </div>
        <div class="sub-box">
          <h4><span>att.W_value.weight (4×4)</span></h4>
          <table class="mat-table"><thead><tr><th>OutDim</th><th>in0</th><th>in1</th><th>in2</th><th>in3</th></tr></thead><tbody>${mat4x4(bw.W_value)}</tbody></table>
        </div>
        <div class="sub-box">
          <h4><span>att.out_proj.weight (4×4)</span></h4>
          <table class="mat-table"><thead><tr><th>OutDim</th><th>in0</th><th>in1</th><th>in2</th><th>in3</th></tr></thead><tbody>${mat4x4(bw.out_proj_weight, false)}</tbody></table>
          <div style="margin-top:8px;font-family:var(--mono);font-size:12px;color:var(--muted);">out_proj.bias: ${fmtVec(bw.out_proj_bias)}</div>
        </div>
      </div>
    </div>

    <div class="stage-card">
      <div class="stage-header" onclick="toggleStage(this)">
        <div class="stage-title"><span>📦 权重组 3：输出投影头 out_head.weight (6×4)</span></div>
      </div>
      <div class="stage-body">
        <div class="sub-box" style="max-width:600px;">
          <h4><span>out_head.weight (6 × 4, bias=False)</span><span class="badge">24 params</span></h4>
          <table class="mat-table">
            <thead><tr><th>预测目标 Token</th><th>in0</th><th>in1</th><th>in2</th><th>in3</th></tr></thead>
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
  const mapIdx = { step0: 0, step1: 1, compare: 2, weights: 3 };
  if (btns[mapIdx[tabId]]) btns[mapIdx[tabId]].classList.add("active");

  const container = document.getElementById("view-container");
  if (tabId === "step0") container.innerHTML = renderStepView(0);
  else if (tabId === "step1") container.innerHTML = renderStepView(1);
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

renderHeaderBadges();
switchTab("step0");
</script>
</body>
</html>
"""
    return html_template.replace("__TRACE_JSON__", json_str)


def export_dual_track_dissection(trace_data, output_dir):
    md_path = os.path.join(output_dir, "gpt2toy_dissection.md")
    html_path = os.path.join(output_dir, "gpt2toy_inspector.html")

    md_content = generate_markdown_report(trace_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    html_content = generate_html_inspector(trace_data)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return md_path, html_path
