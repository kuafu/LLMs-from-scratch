# 提升 LLM 训练速度的 PyTorch 性能技巧



请注意，本书旨在教学，因此原始代码有意保持简单，以提高可读性并确保能在包括 CPU 和 GPU 在内的不同硬件上兼容运行。不过，你可能也想了解一些更高级的 PyTorch 和 GPU 功能，以提升 LLM 的训练性能。

本文件夹包含三个代码文件，演示如何优化第 5 章中介绍的 LLM 和训练函数的性能：

1. [`00_orig.py`](00_orig.py)：第 5 章中用于 CPU 和单 GPU 训练的原始代码。\
   ➤ 运行命令：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：针对单 GPU 训练优化的版本。\
   ➤ 运行命令：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）进行多 GPU 训练的优化版本。\
   ➤ 运行命令：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`\
   （**注意：**为了尽量减少相较于 `01_opt_single_gpu.py` 的改动，此脚本仅支持通过上述 `torchrun` 命令进行多进程运行。这意味着，不能通过 `python 02_opt_multi_gpu_ddp.py` 使用多 GPU。）

**请注意，这些修改将训练速度从每秒 12,525 个词元（单张 A100）提升至每秒 142,156 个词元（单张 A100）和每秒 419,259 个词元（4 张 A100）。**

我计划将来通过更详细的文章介绍这些差异。目前，查看代码新增了哪些改进的最简单方式，是在 Visual Studio Code 中打开这些文件并使用“比较所选内容”功能查看差异。

![VS 比较视图](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch 技巧](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## 单 GPU 速度对比

如上所述，我计划将来进一步介绍这些改动。目前，本节按每项修改列出简单的性能概览，以每秒处理的词元数衡量。所有实验均在 A100 GPU 上运行。

&nbsp;
### 基线

请注意，`00_orig.py` 作为基线，除以下几点外没有重大修改，直接使用了第 5 章中的代码：

- 上下文长度扩大到 4 倍（这解释了与第 5 章代码相比，`00_orig.py` 的内存占用为何相对较大）；
- 批量大小扩大到 4 倍（这是 `00_orig.py` 内存占用相对较大的另一个原因）；
- 使用篇幅更长的公共领域书籍来增加训练数据量。

这些超参数并未针对降低损失和减少过拟合进行充分优化，因此 LLM 最后生成的文本可能不够精致；不过这不影响主要结论，因为此处以 `tok/sec` 指标作为速度参考（数值越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py` 包含下文依次列出的全部修改。

对比始终基于上一节中第一个训练轮次之后的平均 tok/sec 和已分配内存。

&nbsp;
### 1. 即时创建因果掩码

- 不再保存因果掩码，而是在需要时即时创建，以减少内存使用（在这里影响很小，但对于支持 131k 输入词元等长上下文的 Llama 3.2 模型，节省的内存可能会累积起来）

修改前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

修改后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用 TF32 张量核心

- 在 A100 等 Ampere 架构及更新的 GPU 上，为 float32 矩阵乘法启用 TF32。此设置不会影响第 5 步介绍的 bfloat16 矩阵乘法。

修改前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合 AdamW 优化器

- 通过设置 `fused=True` 使用 `AdamW` 的融合内核

修改前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 在数据加载器中使用固定内存

- 在数据加载器中使用 `pin_memory=True`，以预先分配并重复使用 GPU 内存

修改前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 从 32 位浮点精度切换为 16 位 Brain Float（bfloat16）精度（更多信息请参阅我的[这篇文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）
- 请注意，进行 bfloat16 矩阵乘法时，PyTorch 通常会自动使用张量核心（如果 GPU 支持），无需启用 TF32

修改前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

修改后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用 PyTorch 类替换从零实现的代码

- 将从零实现的 LayerNorm 和 GeLU 替换为 PyTorch 的原生实现

修改前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

修改后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 中带有 FlashAttention 的自注意力函数，替代我们从零实现的多头注意力。


修改前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

修改后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。请注意，在速度提升之前，最初几次迭代总是比较慢。由于 `Avg tok/sec` 的测量只纳入平均值计算中的第一行数据，因此现在我们使用第 1 个训练轮次结束时的 `Step tok/sec`。


修改前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

修改后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 说明**

- 在 Windows 上配置编译可能比较棘手。
- `torch.compile()` 使用 Inductor 对内核进行 JIT 编译，因此需要可用的 C/C++ 工具链。
- 对于 CUDA，Inductor 还依赖 Triton；可以通过社区软件包 `triton-windows` 获取。
  - 如果看到 `cl not found`，请[安装包含“C++ 工作负载”的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)，并从“x64 Native Tools”命令提示符运行 Python。
  - 如果使用 CUDA 时看到 `triton not found`，请安装 `triton-windows`（例如，运行 `uv pip install "triton-windows<3.4"`）。
- 对于 CPU，另有读者建议参照这份 [Windows 版 PyTorch Inductor 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)。
  - 请注意，安装 Visual Studio 2022 时应安装英语语言包，以避免 UTF-8 错误。
  - 另外，必须通过“Visual Studio 2022 Developer Command Prompt”运行代码，不能在笔记本中运行。
- 如果此配置过程过于棘手，可以跳过编译；**编译是可选的，所有代码示例都可以在不编译的情况下正常运行**。

---

&nbsp;
### 9. 词汇表填充

- 此处将词汇表大小从 50,257 略微增加到 50,304，也就是最接近的 64 的倍数。这个技巧是我的前同事 Carlos Mocholi 提醒我的；他提到最初的建议来自 Andrej Karpathy（可能是[这篇帖子](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议源于他与 PyTorch 团队的一次交流；[Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29)提到，团队当时就 `torch.compile` 提供了建议。[NVIDIA 关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)是相关的优质资料，其中通常会将批量大小和线性层维度设为某些数值的倍数。此外，NVIDIA 的 Megatron 团队很早就介绍过词汇表填充技巧（参见 2019 年的论文[《Megatron-LM：使用模型并行训练数十亿参数语言模型》](https://arxiv.org/abs/1909.08053)）。

修改前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

修改后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增大批量大小

- 最后，将批量大小增至 GPU 支持的最大 2 的幂。

修改前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

修改后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## 多 GPU 速度对比

这可能并非完全公平的对比，因为现在使用的是 4 个 GPU，而不是 1 个。不过，如果训练不受 GPU 内存不足的限制，使用分布式数据并行这一最快的多 GPU 技术，当然可以显著提升速度：

修改前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

修改后（4 个 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
