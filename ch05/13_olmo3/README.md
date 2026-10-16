# 从零实现 Olmo 3 7B 和 32B

本文件夹中的 Jupyter 笔记本 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) 从零实现了 Olmo 3 7B 和 32B，运行需要约 13 GB RAM。

另一个笔记本 [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) 加入了 KV 缓存，以提升运行时性能（但会增加代码复杂度）。有关 KV 缓存 的更多信息，请参阅我的文章[从零理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

下面是与作为参考模型的 Qwen3 的并列对比；如果你对 Qwen3 0.6B 独立笔记本感兴趣，可以在[此处](../11_qwen3)找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

如下图所示，Olmo 3 还有不同版本（它们的架构相同，只有训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3 与 Qwen3 有何不同

本节聚焦于架构，而非训练细节，简要对比 Olmo 3 和 Qwen3。


7B 模型：

1. 如上图所示，Olmo 3 的架构与 Qwen3 比较相似。不过，值得注意的是，它基本上很可能是受前代 Olmo 2 启发，而不是受 Qwen3 启发。

2）与 Olmo 2 类似，Olmo 3 仍采用后置归一化，而不是前置归一化；Olmo 2 论文发现，这种做法可以稳定训练。

3）有趣的是，7B 模型仍像 Olmo 2 一样使用多头注意力。
不过，为了提高效率并减小 KV 缓存 的大小，现在它使用了滑动窗口注意力（例如，与 Gemma 3 类似）。

接下来是 32B 模型：

4）总体而言，它使用相同的架构，只是规模更大。此外，各项比例（例如，前馈层从输入维度扩展到中间维度的比例等）大致与 Qwen3 相同。

5）我猜测，由于词汇表较小，该架构最初的规模略小于 Qwen3；随后，他们将中间层维度的扩展比例从 Qwen3 的 5 倍提高到 Olmo 3 的 5.4 倍，从而得到一个可直接对比的 32B 模型。

6）另外请注意，32B 模型（终于）使用了分组查询注意力。





<br>

有关架构差异以及与其他架构的对比，请参阅我的文章[《大型 LLM 架构对比：从 DeepSeek-V3 到 Kimi K2，纵览现代 LLM 架构设计》](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。





