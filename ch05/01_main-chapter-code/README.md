# 第 5 章：无标签数据上的预训练

### 章节主要代码

- [ch05.ipynb](ch05.ipynb) 包含本章中展示的全部代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，其中包含前几章中的 `MultiHeadAttention` 模块和 `GPTModel` 类；我们会在 [ch05.ipynb](ch05.ipynb) 中导入它们来预训练 GPT 模型
- [gpt_download.py](gpt_download.py) 包含用于下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章练习的解答

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的 GPT 模型训练代码（可以把它看作总结本章内容的代码文件）
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的代码，用于加载并使用 OpenAI 的预训练模型权重

