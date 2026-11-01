# GPT-2 Toy 全链路深度剖析套件更新说明 (Release Notes)

> **版本**：v2.0 (全链路显微镜与双轨观战升级版)  
> **更新日期**：2026-11-01  
> **适用模块**：[gpt2toy.py](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy.py) 及双轨剖析套件  

---

## 一、 更新背景与目标

在原先的 `gpt2toy.py` 脚本中，虽然输出了基本的模型参数量和部分张量形状，但对于理解 Transformer 底层数学细节仍存在以下痛点：
1. **黑盒运算未拆透**：例如 LayerNorm 的均值与方差如何计算、注意力机制的掩码与点积如何分步演化缺乏同屏比对；
2. **缺乏残差增量视角**：残差连接对主干特征向量改变了多少缺乏直观的三列对照；
3. **因果与自回归机制缺乏跨步对比**：无法直观体会为什么工业界推理能用 KV Cache，也无法看清同一词元在不同位置时的内部表征分化；
4. **载体单一**：终端控制台输出信息密集、难以长期归档或交互式观战。

本次升级针对上述问题进行了**像素级细化**，并构建了**双轨（Markdown 深度报告 + 交互式 HTML 观战面板）全自动导出体系**。

---

## 二、 核心更新内容清单

### 1. 五大微观运行细节补全

| 序号 | 细化维度 | 具体实现与更新点 | 对应代码与文档体现 |
| :---: | :--- | :--- | :--- |
| **1** | **嵌入查表与广播相加透明化** | 将词嵌入查表 `tok_emb[ID]` 与位置嵌入查表 `pos_emb[Pos]` 并排展示，逐元素写出 `tok[d] + pos[d] = x[d]` 的相加映射。 | 终端日志第 1 阶段、Markdown 第 4.1/5.1 节、HTML 阶段一 |
| **2** | **LayerNorm 内部统计量白盒化** | 显式计算并打印每个 Token 的 4 维均值 $\mu$、有偏方差 $\sigma^2$、标准差 $\sqrt{\sigma^2 + 10^{-5}}$，并验证标准化输出满足 $\mu' = 0.0000$ 与 $\sigma'^2 = 1.0000$。 | `LayerNorm.forward`、Markdown 第 4.2/5.2 节、HTML 阶段二 |
| **3** | **多头注意力 (MHA) 全链路显微镜** | ① **切头前后对比**：展示线性层输出的完整 4 维 $Q,K,V$ 与划分到 Head 0 / Head 1 的 $2\times 2$ 空间关系；<br/>② **得分四步演化**：同屏并排展示 `原始点积 (Q@K^T)` $\to$ `缩放得分 (/√2)` $\to$ `因果掩码遮蔽 (-inf)` $\to$ `Softmax权重`；<br/>③ **上下文聚合**：展示各头加权求和、左右拼接及 `out_proj` 投影。 | `MultiHeadAttention.forward`、Markdown 第 4.3/5.3 节、HTML 阶段三 |
| **4** | **残差连接 (Shortcut) 三列增量对照** | 将残差连接展示为 `shortcut (主干输入)` + `sublayer_out (MHA/FFN输出)` = `x_updated (融合输出)`，看清每个模块对特征向量的微调增量。 | `print_residual_comparison`、Markdown 第 4.4/5.4 节、HTML 阶段四/五 |
| **5** | **跨步因果不变性与同词异位深度对比** | ① **KV Cache 原理实证**：实测验证 Step 1 ($T=4$) 与 Step 2 ($T=5$) 前 4 个位置的 15 组张量最大误差恒为 `0.00000000`，证明因果掩码对历史信息的隔离；<br/>② **同词异位对比**：比对 Step 2 中同为 `'I' (ID=4)` 的 $t_2$ 与 $t_4$，展示位置编码与因果视野如何促使模型做出不同的后继词预测。 | `analyze_and_print_cross_step_insights`、Markdown 第 6 节、HTML 跨步对比 Tab |

---

### 2. 双轨自动化导出体系

运行 `python gpt2toy.py` 会在执行终端日志打印的同时，自动将录制的数据快照导出为两种载体：

#### 轨道一：Markdown 深度剖析报告 ([gpt2toy_dissection.md](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy_dissection.md))
* **架构演化流**：内嵌完整的 Mermaid 流程拓扑图，标明各节点形状变化；
* **参数与权重台账**：详细拆解 312 个可学习参数，提供初始权重完整表格（包含可折叠展开的 FFN 权重）；
* **数学与表格对照**：全量配备 LaTeX 数学定义与实际 PyTorch 张量运行结果；
* **离线归档友好**：非常适合在 VS Code、Obsidian、GitHub 中作为技术笔记沉淀。

#### 轨道二：单文件交互式 HTML 观战面板 ([gpt2toy_inspector.html](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy_inspector.html))
* **零外部依赖**：单 HTML 文件（无需联网下载外部 JS/CSS），直接双击即可用任意现代浏览器查看；
* **Tab 自由切换**：支持在 `Step 1 (T=4)`、`Step 2 (T=5)`、`跨步对比 (KV Cache & 同词异位)`、`初始权重总览` 之间秒切；
* **智能交互与色彩热力图**：
  * 注意力权重矩阵与 Softmax 分布自带颜色深浅渲染；
  * **鼠标悬停验算（Tooltip）**：鼠标移动至任一单元格（如注意力得分、LayerNorm 结果、词嵌入融合），自动弹出其底层计算来源与验算公式；
  * 一键展开/折叠全部计算阶段。

---

## 三、 生成文件与使用方式

### 1. 文件结构
```text
gpt2toy/
├── gpt2toy.py              # 主脚本 (包含微型 GPT-2 架构、执行流与快照录制器)
├── gpt2toy_exporter.py     # 双轨导出器 (将运行时快照转换为 Markdown 与 HTML)
├── gpt2toy_dissection.md   # [产物] 格式化图文 Markdown 深度剖析报告
├── gpt2toy_inspector.html  # [产物] 单文件交互式观战面板
└── UPDATE_NOTES.md         # [本文档] 本次升级与细节剖析更新说明
```

### 2. 运行与查看方法

1. **重新执行剖析并导出**：
   ```powershell
   python gpt2toy.py
   ```
2. **阅读静态报告**：
   * 在 VS Code 或编辑器中预览 [gpt2toy_dissection.md](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy_dissection.md)。
3. **在浏览器中交互观战**：
   * 双击打开 [gpt2toy_inspector.html](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2toy/gpt2toy_inspector.html)；
   * 在顶部切换 Step，鼠标悬停于张量数值单元格查看实时验算公式。
