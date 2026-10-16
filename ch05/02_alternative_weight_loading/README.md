# 加载预训练权重的其他方法

如果无法从 OpenAI 获取权重，本文件夹提供了其他权重加载方案。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：（推荐）包含从 PyTorch state dict 加载权重的代码；这些 state dict 是我通过转换原始 TensorFlow 权重创建的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过 `transformers` 库从 Hugging Face Model Hub 加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含直接通过 `safetensors` 库从 Hugging Face Model Hub 加载权重的代码（跳过实例化 Hugging Face transformer 模型这一步）
