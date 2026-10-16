# Gemma 4

此目录包含一个独立的纯文本 Gemma 4 笔记本，它以 Gemma 3 参考笔记本为基础构建，并适配了稠密型 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` 检查点。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) 使用纯 PyTorch 实现 Gemma 4 共有的稠密架构，并通过 `CHOOSE_MODEL` 在 E2B 和 E4B 参考配置之间切换。
