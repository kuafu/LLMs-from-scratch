# 从零实现 Qwen3.5 0.8B

本文件夹包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从零实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在文章[《超越标准 LLM》](https://magazine.sebastianraschka.com/p/beyond-standard-llms)的[第 2 节“（线性）注意力混合架构”](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids)中对此作了更详细的介绍。

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

请注意，Qwen3.5 交替使用 `linear_attention` 层和 `full_attention` 层。\
这些笔记本在保持完整模型流程清晰易读的同时，复用了 [qwen3_5_transformers.py](qwen3_5_transformers.py) 中的线性注意力构件。该文件包含来自 Hugging Face 的线性注意力代码，并遵循 Apache 2.0 开源许可证。

&nbsp;
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb)：Qwen3.5 0.8B 的主要笔记本实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb)：加入 KV-cache 解码以提高效率的同一模型版本。
- [qwen3_5_transformers.py](qwen3_5_transformers.py)：用于 Qwen3.5 线性注意力的部分 Hugging Face Transformers 辅助组件。

