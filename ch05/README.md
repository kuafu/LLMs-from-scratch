# 第 5 章：无标签数据上的预训练

&nbsp;
## 章节主要代码

- [01_main-chapter-code](01_main-chapter-code) 包含本章的主要代码

&nbsp;
## 附加材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含在无法从 OpenAI 获取模型权重时，从其他来源加载 GPT 模型权重的代码
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含使用 Project Gutenberg 的整套图书语料对 LLM 进行更长时间预训练的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含更完善的训练函数实现，其中加入了学习率调度器和梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含一个可选的超参数调优脚本
- [06_user_interface](06_user_interface) 实现了一个可与预训练 LLM 交互的用户界面
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个附加笔记本，演示如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含 GPT-2 BPE 分词器的从零实现
- [10_llm-training-speed](10_llm-training-speed) 展示了可提升 LLM 训练速度的 PyTorch 性能技巧
- [18_muon](18_muon) 介绍如何在 GPT 模型训练设置中使用 Muon 优化器
- [19_cpu_mps_differences](19_cpu_mps_differences) 汇集了 CPU 与 MPS 设备行为不同的示例

&nbsp;
<a id="llm-architectures-from-scratch"></a>
## 从零构建 LLM 架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;


- [07_gpt_to_llama](07_gpt_to_llama) 包含逐步将 GPT 架构实现转换为 Llama 3.2 的指南，并加载 Meta AI 的预训练权重
- [11_qwen3](11_qwen3) 从零实现了 Qwen3 0.6B 和 Qwen3 30B-A3B（混合专家）模型，并包含加载基础版、推理版和代码版模型变体预训练权重的代码
- [12_gemma3](12_gemma3) 从零实现了 Gemma 3 270M，并提供带 KV 缓存 的变体，其中包含加载预训练权重的代码
- [13_olmo3](13_olmo3) 从零实现了 Olmo 3 7B 和 32B（Base、Instruct 和 Think 变体），并提供带 KV 缓存 的变体，其中包含加载预训练权重的代码
- [17_gemma4](17_gemma4) 从零实现了 Gemma 4 的 E2B 和 E4B 稠密变体

&nbsp;
## 本章跟做视频

<br>
<br>

[![视频链接](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
