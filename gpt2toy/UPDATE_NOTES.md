# GPT-2 Toy 全链路深度剖析套件更新说明 (Release Notes)

> **版本**：v3.1 (默认上下文长度调整为 5)
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
| **2** | **LayerNorm 内部统计量白盒化** | 展示每个 Token 沿特征维计算的均值 $\mu$、有偏方差 $\sigma^2$、标准差及仿射变换后输出的实测均值与方差；不把学习后的输出统计量预设为 0 和 1。 | `LayerNorm.forward`、Markdown 第 4.2/5.2 节、HTML 阶段二 |
| **3** | **多头注意力 (MHA) 全链路显微镜** | ① **切头前后对比**：展示线性层输出的完整 4 维 $Q,K,V$ 与划分到 Head 0 / Head 1 的 $2\times 2$ 空间关系；<br/>② **得分四步演化**：同屏并排展示 `原始点积 (Q@K^T)` $\to$ `缩放得分 (/√2)` $\to$ `因果掩码遮蔽 (-inf)` $\to$ `Softmax权重`；<br/>③ **上下文聚合**：展示各头加权求和、左右拼接及 `out_proj` 投影。 | `MultiHeadAttention.forward`、Markdown 第 4.3/5.3 节、HTML 阶段三 |
| **4** | **残差连接 (Shortcut) 三列增量对照** | 将残差连接展示为 `shortcut (主干输入)` + `sublayer_out (MHA/FFN输出)` = `x_updated (融合输出)`，看清每个模块对特征向量的微调增量。 | `print_residual_comparison`、Markdown 第 4.4/5.4 节、HTML 阶段四/五 |
| **5** | **跨步因果前缀与同词异位对照** | Step 1 ($T=4$) 与 Step 2 ($T=5$) 只在模型处于 eval 模式、输入前缀与位置对齐时比较共同位置；最大绝对误差不超过 `1e-6` 标为 `verified`，超过则标为 `different`。若窗口裁剪导致 token 或位置改变，标为 `not_comparable`，不把错位结果解释成因果差异。独立输入 `Hello, I am I` 的 IDs 为 `[2, 3, 4, 5, 4]`，其中 `I` 位于位置 2 和 4。当前实测两处 logits 不同、argmax 却都为 `am` (ID 5)；分数变化不一定改变预测词，所有结论以本次张量为准。 | `analyze_and_print_cross_step_insights`、Markdown 第 6 节、HTML 跨步对比 Tab |

---

### 2. 双轨自动化导出体系

运行 `python run_gpt2toy.py` 会在执行终端日志打印的同时，将日志保存到 `gpt2toy.txt`，并自动将录制的数据快照导出为两种载体：

#### 轨道一：Markdown 深度剖析报告 ([gpt2toy_dissection.md](gpt2toy_dissection.md))
* **架构演化流**：内嵌完整的 Mermaid 流程拓扑图，标明各节点形状变化；
* **参数与权重台账**：参数数量和权重表从本次模型实例采集；当前默认配置有 308 个可学习参数，位置嵌入为 `[5, 4]`；
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
├── test_gpt2toy_teaching.py # 上下文配置、固定重复词输入、比较结论与导出测试
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

from run_gpt2toy import REPEATED_CONTEXT, SEED, START_CONTEXT, TOY_CONFIG, TOY_VOCAB
from gpt2toy_data import SimpleTokenizerV2

torch.manual_seed(SEED)
model = GPTModel(TOY_CONFIG).eval()
tokenizer = SimpleTokenizerV2(TOY_VOCAB)
ids = torch.tensor([tokenizer.encode(START_CONTEXT)])  # [2, 3, 4, 5]
repeated_ids = tokenizer.encode(REPEATED_CONTEXT)       # [2, 3, 4, 5, 4]

logits = model(ids)  # 无打印、快照、CPU 转换或导出。
out = generate_text_simple(
    model, ids, max_new_tokens=2, context_size=TOY_CONFIG["context_length"]
)

trace = TeachingTrace(verbose=False)
traced_logits = model(ids, trace=trace)
assert torch.equal(logits, traced_logits)
attention = trace.record["blocks"][0]["att"]
```

`TeachingTrace` 接收算法在原计算位置产生的张量，同步生成独立的 Python 数值快照，
不保存模型或计算图；缩放前得分、归一化后统计量和展示用概率均在诊断模块中处理。
教学快照记录首个 batch 样本，`verbose` 控制打印，`show_data` 控制详细数值展示。
生成过程的教学打印还需传入 `tokenizer`，完整示例见 `run_gpt2toy.py`。

启用教学打印时，`GPTModel`、`TransformerBlock`、`MultiHeadAttention`、`LayerNorm`、
`FeedForward` 和 `GELU` 的 `forward` 都打印 `begin/end` 与输入/输出形状。标记按实际调用
深度每层缩进 4 个空格：Block 内的 LN/MHA/FFN 是同级调用，GELU 是 FFN 的子调用，
其输入形状为升维后的 `[B,T,4D]`；最终 LN 则是 GPTModel 的直接子调用。
独立调用这些模块时从根层级开始，同一观测器可顺序复用。

原来的 `verbose`、`show_data` 和 `record_dict` 模型参数已由可选 `trace` 替代。
自定义回调遵循 `trace(event, **values)`，同步只读观察张量；不要修改张量或消耗随机数。
同一个 `TeachingTrace` 可顺序复用；并发执行应分别创建实例。
当前默认使用 `SEED = 123`、`context_length = 5`、`START_CONTEXT = "Hello, I am"` 与独立重复词输入 `REPEATED_CONTEXT = "Hello, I am I"`。两步生成从 IDs `[2, 3, 4, 5]` 开始，输入长度依次为 4、5，最终序列长度为 6。当前 CPU 默认初始化的示例结果是 IDs `[2, 3, 4, 5, 1, 5]`，文本 `Hello, I am <|endoftext|> am`。生成器不会因为产生 `<|endoftext|>` 而提前停止。

同一个 seed 只固定同一配置下的初始化过程。把上下文嵌入从 `[6, 4]` 改成 `[5, 4]` 会改变参数形状及后续随机数消耗，因此新模型有 308 个参数，且本次权重与旧默认模型不同；不能用旧输出当作当前输出。

需要将演示输出写入其他目录时，可调用 `run_gpt2toy.run(output_dir=...)`。
默认输出目录仍为脚本所在目录，每次运行覆盖日志及两份导出文件。

运行回归测试：

```powershell
python -B -m unittest test_gpt2toy test_gpt2toy_teaching -v
```

当前 v3.1 已完成以下 CPU 验证：

- 14 项回归测试通过；包含关闭 trace 的静默张量路径、Dropout 训练的输出与梯度一致性、
  固定重复词样本的独立前向、比较成功 / 失败 / 不可比较，以及另一组模型尺寸的报告。
  新增用实际模块调用顺序核对 begin/end 嵌套、缩进、形状和独立子模块调用的测试。
- 在相同的长度 5 配置和权重下，对照改动前的核心实现，eval 与带 Dropout 的训练场景中，
  logits、全部参数梯度和随机数状态完全一致。默认模型容量变化导致的新初始化结果单独记录。
- 在实际 Chromium 浏览器中验证了默认配置、另一组尺寸、多步裁剪、单步、零步及比较失败共六种报告；
  无 JavaScript 错误，步骤切换、矩阵表头、折叠与悬浮提示正常。
- 独立运行 `python run_gpt2toy.py` 已更新 UTF-8 日志、Markdown 报告及 HTML 面板；日志与终端内容一致。
  相关架构图的 SVG、HTML 和 PNG 也已按当前模型重新生成。

以下是 **v3.0 重构的历史验证记录**，使用旧默认上下文长度 6；它只说明当时新旧实现之间的 parity，不代表当前 `context_length=5` 的输出或验收状态：

- 四组配置分别开启、关闭 trace，共八组对照；涵盖多 batch、多层、QKV 偏置、零层和带 Dropout 的训练。
  初始化参数、严格加载的 `state_dict`、前向结果、全部参数梯度与随机数状态均完全一致，最大输出误差为 0。
- 当时默认演示的完整快照数据相同，最终 Token IDs 为 `[2, 3, 4, 5, 4, 4]`；此结果对应旧配置，不是当前预期输出。
  587 行终端输出仅输出目录路径不同，日志与终端内容相同，Markdown / HTML 导出逐字节一致。
- 七项回归测试通过，覆盖无 trace 时的静默张量路径、训练观测隔离、快照复用、因果前缀、
  batch 生成与窗口裁剪、分词和数据窗口，以及单头 / 三头的完整教学打印。
