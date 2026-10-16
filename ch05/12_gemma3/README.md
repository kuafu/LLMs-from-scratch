# 从零实现 Gemma 3 270M

本文件夹中的 Jupyter 笔记本 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) 从零实现了 Gemma 3 270M，运行需要约 2 GB RAM。

另一个笔记本 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) 加入了 KV 缓存，以提升运行时性能（但会增加代码复杂度）。有关 KV 缓存 的更多信息，请参阅我的文章[从零理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

| 模型              | 模式              | 硬件            | 令牌/秒    | GPU 内存（VRAM）   |
| ----------------- | ----------------- | --------------- | ---------- | ----------------- |
| Gemma3Model 270M  | 普通              | Mac Mini M4 CPU | 8          | -                 |
| Gemma3Model 270M  | 普通（已编译）    | Mac Mini M4 CPU | 9          | -                 |
| Gemma3Model 270M  | KV 缓存          | Mac Mini M4 CPU | 130        | -                 |
| Gemma3Model 270M  | KV 缓存（已编译）| Mac Mini M4 CPU | 224        | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | 普通              | Mac Mini M4 GPU | 16         | -                 |
| Gemma3Model 270M  | 普通（已编译）    | Mac Mini M4 GPU | 错误       | -                 |
| Gemma3Model 270M  | KV 缓存          | Mac Mini M4 GPU | 23         | -                 |
| Gemma3Model 270M  | KV 缓存（已编译）| Mac Mini M4 GPU | 错误       | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | 普通              | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M  | 普通（已编译）    | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M  | KV 缓存          | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M  | KV 缓存（已编译）| Nvidia A100 GPU | 99         | 2.12 GB           |


下面是与作为参考模型的 Qwen3 0.6B 的并列对比；如果你对 Qwen3 0.6B 独立笔记本感兴趣，可以在[此处](../11_qwen3)找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

有关架构差异以及与其他架构的对比，请参阅我的文章[《大型 LLM 架构对比：从 DeepSeek-V3 到 Kimi K2，纵览现代 LLM 架构设计》](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。





