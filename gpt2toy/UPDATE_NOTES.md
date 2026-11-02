# GPT-2 Toy 全链路深度剖析套件更新说明 (Release Notes)

> **版本**：v3.0 (核心算法与教学诊断分离)
>
> **更新日期**：2026-11-02
>
> **适用模块**：[gpt2toy.py](gpt2toy.py)、[run_gpt2toy.py](run_gpt2toy.py) 及双轨剖析套件

核心算法集中在 `gpt2toy.py`；完整教学演示的运行入口改为 `python run_gpt2toy.py`。
分词与数据窗口、教学打印与快照、文件导出各自独立。以下教学功能均保留。

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

运行 `python run_gpt2toy.py` 会在执行终端日志打印的同时，将日志保存到 `gpt2toy.txt`，并自动将录制的数据快照导出为两种载体：

#### 轨道一：Markdown 深度剖析报告 ([gpt2toy_dissection.md](gpt2toy_dissection.md))
* **架构演化流**：内嵌完整的 Mermaid 流程拓扑图，标明各节点形状变化；
* **参数与权重台账**：详细拆解 312 个可学习参数，提供初始权重完整表格（包含可折叠展开的 FFN 权重）；
* **数学与表格对照**：全量配备 LaTeX 数学定义与实际 PyTorch 张量运行结果；
* **离线归档友好**：非常适合在 VS Code、Obsidian、GitHub 中作为技术笔记沉淀。

#### 轨道二：单文件交互式 HTML 观战面板 ([gpt2toy_inspector.html](gpt2toy_inspector.html))
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
├── gpt2toy.py              # GPT 核心算法与少量可选 trace 回调
├── gpt2toy_data.py         # 分词器、Dataset 与 DataLoader
├── gpt2toy_diagnostics.py  # 教学打印、快照、参数统计、跨步剖析和日志工具
├── gpt2toy_exporter.py     # 将运行时快照转换为 Markdown 与 HTML
├── run_gpt2toy.py          # 完整演示入口：词表、配置、运行与输出文件
├── test_gpt2toy.py         # 核心算法与诊断隔离的回归测试
├── gpt2toy.txt             # [产物] UTF-8 终端日志
├── gpt2toy_dissection.md   # [产物] Markdown 深度剖析报告
├── gpt2toy_inspector.html  # [产物] 单文件交互式观战面板
└── UPDATE_NOTES.md         # [本文档] 使用与更新说明
```

### 2. 运行与查看方法

1. **重新执行剖析并导出**：
   ```powershell
   python run_gpt2toy.py
   ```
2. **阅读静态报告**：
   * 在 VS Code 或编辑器中预览 [gpt2toy_dissection.md](gpt2toy_dissection.md)。
3. **在浏览器中交互观战**：
   * 双击打开 [gpt2toy_inspector.html](gpt2toy_inspector.html)；
   * 在顶部切换 Step，鼠标悬停于张量数值单元格查看实时验算公式。

## 四、核心算法与可选 trace

`gpt2toy.py` 仅依赖 PyTorch，不导入数据、诊断或导出模块，也不调整终端编码。
不传入 trace 时，前向传播只返回 logits；生成函数只接受并返回 Token ID 张量，支持整个 batch。

```python
import torch
from gpt2toy import GPTModel, generate_text_simple
from gpt2toy_diagnostics import TeachingTrace

cfg = dict(vocab_size=6, context_length=6, emb_dim=4,
           n_heads=2, n_layers=1, drop_rate=0.0, qkv_bias=False)
torch.manual_seed(123)
model = GPTModel(cfg).eval()
ids = torch.tensor([[2, 3, 4, 5]])

logits = model(ids)  # 无打印、快照、CPU 转换或导出。
out = generate_text_simple(model, ids, max_new_tokens=2, context_size=6)

trace = TeachingTrace(verbose=False)
traced_logits = model(ids, trace=trace)
assert torch.equal(logits, traced_logits)
attention = trace.record["blocks"][0]["att"]
```

`TeachingTrace` 接收算法在原计算位置产生的张量，同步生成独立的 Python 数值快照，
不保存模型或计算图；缩放前得分、归一化后统计量和展示用概率均在诊断模块中处理。
教学快照记录首个 batch 样本，`verbose` 控制打印，`show_data` 控制详细数值展示。
生成过程的教学打印还需传入 `tokenizer`，完整示例见 `run_gpt2toy.py`。

原来的 `verbose`、`show_data` 和 `record_dict` 模型参数已由可选 `trace` 替代。
自定义回调遵循 `trace(event, **values)`，同步只读观察张量；不要修改张量或消耗随机数。
同一个 `TeachingTrace` 可顺序复用；并发执行应分别创建实例。
跨步因果与同词异位的讲解继续使用默认微型词表与两步生成示例。

需要将演示输出写入其他目录时，可调用 `run_gpt2toy.run(output_dir=...)`。
默认输出目录仍为脚本所在目录，每次运行覆盖日志及两份导出文件。

运行回归测试：

```powershell
python -B -m unittest test_gpt2toy -v
```

本次重构在 CPU / PyTorch 2.14.1 上与保留的旧实现做了对照：

- 四组配置分别开启、关闭 trace，共八组对照；涵盖多 batch、多层、QKV 偏置、零层和带 Dropout 的训练。
  初始化参数、严格加载的 `state_dict`、前向结果、全部参数梯度与随机数状态均完全一致，最大输出误差为 0。
- 默认演示的完整快照数据相同，最终 Token IDs 为 `[2, 3, 4, 5, 4, 4]`。
  587 行终端输出仅输出目录路径不同，日志与终端内容相同，Markdown / HTML 导出逐字节一致。
- 七项回归测试通过，覆盖无 trace 时的静默张量路径、训练观测隔离、快照复用、因果前缀、
  batch 生成与窗口裁剪、分词和数据窗口，以及单头 / 三头的完整教学打印。
