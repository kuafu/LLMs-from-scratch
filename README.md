# 从零构建大语言模型

本仓库包含开发、预训练和微调类 GPT 大语言模型的代码，也是《[从零构建大语言模型（Build a Large Language Model (From Scratch)）](https://amzn.to/4fqvn0D)》一书的官方代码仓库。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在《[*从零构建大语言模型*](http://mng.bz/orYv)》中，你将从基础开始，一步步编写代码，由内而外地理解大语言模型（LLM）的工作原理。我会带你创建自己的 LLM，并通过清晰的文字、图示和示例讲解各个阶段。

本书介绍如何出于学习目的，开发并训练一个规模较小但功能完整的模型。这套方法与构建 ChatGPT 背后那类大规模基础模型所用的方法一致。此外，本书还提供了加载更大型预训练模型权重并进行微调的代码。

- 官方[源代码仓库](https://github.com/rasbt/LLMs-from-scratch)
- [Manning 出版社官网上的图书页面](http://mng.bz/orYv)
- [Amazon.com 上的图书页面](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载本仓库，请点击[下载 ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip)，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果你从 Manning 网站下载了代码包，建议访问 GitHub 上的官方代码仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，获取最新更新。）

<br>
<br>


# 目录

请注意，`README.md` 是 Markdown（`.md`）文件。如果你从 Manning 网站下载了代码包，并在本地计算机上查看，建议使用 Markdown 编辑器或预览工具，以获得正确的显示效果。如果尚未安装 Markdown 编辑器，[Ghostwriter](https://ghostwriter.kde.org) 是一个不错的免费选择。

你也可以在浏览器中访问 GitHub 仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，查看本文件及其他文件；GitHub 会自动渲染 Markdown。

<br>
<br>


> **提示：**
> 如果你需要安装 Python、安装 Python 包以及配置代码运行环境的指导，建议阅读 [setup](setup) 目录中的 [README.md](setup/README.md)。

<br>
<br>

[![Linux 代码测试](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Windows 代码测试](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![macOS 代码测试](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

- [故障排查指南](./troubleshooting.md)
- [LLM 术语与缩写速查](./LLM术语与缩写速查.md)
- [Python 对象调用与源码定位参考](./Python对象调用与源码定位参考.md)


| 章节标题                                                   | 主要代码（快速访问）                                                                                                             | 全部代码与补充资料             |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [环境配置建议](setup) <br/>[如何更好地阅读本书](https://sebastianraschka.com/blog/2025/reading-books.html) | - | - |
| 第 1 章：理解大语言模型 | 无代码 | - |
| 第 2 章：处理文本数据 | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb)（概要）<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02) |
| 第 3 章：编写注意力机制代码 | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb)（概要）<br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb) | [./ch03](./ch03) |
| 第 4 章：从零实现 GPT 模型 | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py)（概要）<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04) |
| 第 5 章：在无标签数据上预训练 | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py)（概要）<br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py)（概要）<br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05) |
| 第 6 章：面向文本分类的微调 | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)<br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)<br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06) |
| 第 7 章：面向指令遵循的微调 | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py)（概要）<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py)（概要）<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07) |
| 附录 A：PyTorch 入门 | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 附录 B：参考文献与延伸阅读 | 无代码 | [./appendix-B](./appendix-B) |
| 附录 C：练习解答 | - [练习解答列表](appendix-C) | [./appendix-C](./appendix-C) |
| 附录 D：完善训练循环 | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb) | [./appendix-D](./appendix-D) |
| 附录 E：使用 LoRA 进行参数高效微调 | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb) | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下图概括了本书涵盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## 前置知识

最重要的前提是具备扎实的 Python 编程基础。
有了这些知识，你就能充分准备好探索精彩的 LLM 世界，
并理解本书介绍的概念和代码示例。

如果你有深度神经网络方面的经验，某些概念可能会更容易理解，因为 LLM 正是基于这些架构构建的。

本书使用 PyTorch 从零实现代码，不依赖外部 LLM 库。你不需要预先精通 PyTorch，但了解其基础知识会有所帮助。如果你刚接触 PyTorch，附录 A 提供了一份简明入门介绍。你也可以阅读我的另一本书《[一小时学会 PyTorch：从张量到多 GPU 神经网络训练](https://sebastianraschka.com/teaching/pytorch-1h/)》，学习其中的核心知识。



<br>
&nbsp;

## 硬件要求

本书主要章节的代码可以在普通笔记本电脑上于合理时间内运行，无需专用硬件，因此更多读者都能动手学习。此外，如果有可用的 GPU，代码会自动使用它。（更多建议请参阅[环境配置文档](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md)。）


&nbsp;
## 视频课程

[配套视频课程总长 17 小时 15 分钟](https://www.manning.com/livevideo/master-and-build-large-language-models)，由我逐章演示本书代码的编写过程。课程的章、节安排与本书结构对应，既可以独立学习，也可以作为配套编程资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## 配套图书与续作

《[*从零构建推理模型（Build A Reasoning Model (From Scratch)）*](https://mng.bz/lZ5B)》是一本可独立阅读的图书，也可以视为《从零构建大语言模型》的续作。

它以预训练模型为起点，实现推理时扩展、强化学习和蒸馏等不同方法，以提高模型的推理能力。

与《从零构建大语言模型》类似，《[*从零构建推理模型*](https://mng.bz/lZ5B)》同样注重实践，带你从零实现这些方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- [Amazon 页面](https://amzn.to/4aAKiFY)
- [Manning 页面](https://mng.bz/lZ5B)
- [GitHub 仓库](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## 练习

本书每章都包含若干练习。解答汇总在附录 C 中，对应的代码 Notebook 位于本仓库各章的主代码目录下，例如 [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)。

除了代码练习，你还可以从 Manning 网站免费下载一份 170 页的 PDF《[从零构建大语言模型：自测题集](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)》。它为每章提供约 30 道测验题及解答，帮助你检验理解程度。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## 补充资料

以下目录提供可选的补充资料，供感兴趣的读者学习：
- **环境配置**
  - [Python 环境配置建议](setup/01_optional-python-setup-preferences)
  - [安装本书使用的 Python 包与库](setup/02_installing-python-libraries)
  - [Docker 环境配置指南](setup/03_optional-docker-environment)

- **第 2 章：处理文本数据**
  - [从零实现字节对编码（BPE）分词器](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [比较不同的字节对编码（BPE）实现](ch02/02_bonus_bytepair-encoder)
  - [理解嵌入层与线性层的区别](ch02/03_bonus_embedding-vs-matmul)
  - [用简单数字直观理解数据加载器](ch02/04_bonus_dataloader-intuition)
  - [从零实现 BPE](ch02/05_bpe-from-scratch)
  - [SimpleTokenizerV3 变体](ch02/06_bonus_simple-tokenizer-v3)

- **第 3 章：编写注意力机制代码**
  - [比较高效多头注意力的不同实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解 PyTorch 缓冲区](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **第 4 章：从零实现 GPT 模型**
  - [浮点运算量（FLOPs）分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 缓存](ch04/03_kv-cache)
  - [注意力机制的替代方案](ch04/#attention-alternatives)
    - [分组查询注意力](ch04/04_gqa)
    - [多头潜在注意力](ch04/05_mla)
    - [滑动窗口注意力](ch04/06_swa)
    - [门控 DeltaNet](ch04/08_deltanet)
    - [DeepSeek 稀疏注意力（DSA）](ch04/09_dsa)
    - [跨层 KV 共享](ch04/10_kv-sharing)
  - [混合专家（MoE）](ch04/07_moe)

- **第 5 章：在无标签数据上预训练**
  - [其他权重加载方法](ch05/02_alternative_weight_loading/)
  - [在 Project Gutenberg 数据集上预训练 GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [完善训练循环](ch05/04_learning_rate_schedulers)
  - [CPU 与 MPS 设备之间的差异](ch05/19_cpu_mps_differences)
  - [优化预训练超参数](ch05/05_bonus_hparam_tuning)
  - [构建与预训练 LLM 交互的用户界面](ch05/06_user_interface)
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [节省内存的模型权重加载](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [为 Tiktoken BPE 分词器扩展新词元](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [加速 LLM 训练的 PyTorch 性能优化建议](ch05/10_llm-training-speed)
  - [LLM 架构](ch05/#llm-architectures-from-scratch)
    - [从零实现 Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [从零实现 Qwen3 稠密模型与混合专家（MoE）模型](ch05/11_qwen3/)
    - [从零实现 Gemma 3](ch05/12_gemma3/)
    - [从零实现 Olmo 3](ch05/13_olmo3/)
    - [从零实现 Tiny Aya](ch05/15_tiny-aya/)
    - [从零实现 Qwen3.5](ch05/16_qwen3.5/)
    - [从零实现 Gemma 4 E2B 与 E4B](ch05/17_gemma4/)
  - [在第 5 章中直接替换为其他 LLM（如 Llama 3、Qwen 3）](ch05/14_ch05_with_other_llms/)
- **第 6 章：面向分类任务的微调**
  - [微调不同层和使用更大模型的补充实验](ch06/02_bonus_additional-experiments)
  - [在含 5 万条 IMDb 影评的数据集上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建与基于 GPT 的垃圾信息分类器交互的用户界面](ch06/04_user_interface)
- **第 7 章：面向指令遵循的微调**
  - [查找近似重复项和生成被动语态条目的数据集工具](ch07/02_dataset-utilities)
  - [使用 OpenAI API 和 Ollama 评估指令回复](ch07/03_model-evaluation)
  - [生成用于指令微调的数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进用于指令微调的数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 和 Ollama 生成偏好数据集](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [用于 LLM 对齐的直接偏好优化（DPO）](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建与指令微调后的 GPT 模型交互的用户界面](ch07/06_user_interface)

来自 [Reasoning From Scratch（从零构建推理模型）](https://github.com/rasbt/reasoning-from-scratch) 仓库的更多补充资料：

- **Qwen3（从零实现）基础**
  - [Qwen3 源代码讲解](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [优化版 Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **评估**
  - [基于验证器的评估（MATH-500）](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [多项选择题评估（MMLU）](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM 排行榜评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [以 LLM 为评判者的评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **推理扩展**
  - [自一致性](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [自我改进](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **强化学习（RL）**
  - [从零实现使用 GRPO 的可验证奖励强化学习（RLVR）](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
&nbsp;

## 问题、反馈与仓库贡献


欢迎提出各种反馈，最好通过 [Manning 论坛](https://livebook.manning.com/forum?product=raschka&page=1)或 [GitHub 讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)分享。如果你有问题，或只是想与他人交流想法，也欢迎在论坛发帖。

请注意，本仓库的代码与纸质书对应，因此目前无法接受扩展主要章节代码内容的贡献，否则会导致代码与纸质书出现差异。保持两者一致，有助于所有读者顺畅学习。


&nbsp;
## 引用

如果本书或代码对你的研究有所帮助，欢迎引用。

芝加哥格式引用：

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX 条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
