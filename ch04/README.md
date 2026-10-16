# 第 4 章：从零开始实现 GPT 模型以生成文本

&nbsp;
## 主要章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码。

&nbsp;
## 扩展材料

- [02_performance-analysis](02_performance-analysis) 包含可选代码，用于分析本章实现的 GPT 模型的性能
- [03_kv-cache](03_kv-cache) 实现 KV 缓存，以加快推理期间的文本生成
- [07_moe](07_moe) 介绍并实现专家混合（MoE）
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 提供分步指南，演示如何将 GPT 架构实现转换为 Llama 3.2，并加载 Meta AI 的预训练权重（完成第 4 章后，可以看看其他架构；也可以等读完第 5 章再看）


&nbsp;
<a id="attention-alternatives"></a>
## 注意力机制的替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 介绍分组查询注意力（GQA）。许多现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）都使用它作为常规多头注意力（MHA）的替代方案
- [05_mla](05_mla) 介绍多头潜在注意力（MLA）。DeepSeek V3 使用它作为常规多头注意力（MHA）的替代方案
- [06_swa](06_swa) 介绍滑动窗口注意力（SWA），Gemma 3 等模型使用了这种方法
- [08_deltanet](08_deltanet) 介绍门控 DeltaNet，这是一种常见的线性注意力变体（用于 Qwen3-Next 和 Kimi Linear）
- [10_kv-sharing](10_kv-sharing) 介绍跨层 KV 共享。Gemma 4 E2B 和 E4B 使用它来减少 KV 缓存的内存占用


&nbsp;
## 更多内容

下面的视频提供了一段跟着编写代码的课程，作为补充材料讲解本章的部分内容。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
