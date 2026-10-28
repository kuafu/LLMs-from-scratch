# 第 4 章：从零开始实现 GPT 模型以生成文本

### 主要章节代码

- [ch04.ipynb](ch04.ipynb)：4.1–4.5 节，介绍 GPT 架构、层归一化、前馈网络、捷径连接和 Transformer 块
- [ch04_4.6.ipynb](ch04_4.6.ipynb)：4.6 节，实现 GPT 模型
- [ch04_4.7.ipynb](ch04_4.7.ipynb)：4.7 节，生成文本，包含本章总结
- [previous_chapters.py](previous_chapters.py) 包含上一章的 `MultiHeadAttention`，供 4.5 节的 Transformer 块使用；4.6 和 4.7 节可独立运行，使用 [gpt.py](gpt.py) 中的相关组件

### 可选代码

- [gpt.py](gpt.py) 是独立的 Python 脚本，包含目前为止实现的代码，包括本章编写的 GPT 模型

