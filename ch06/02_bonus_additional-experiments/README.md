# 更多分类微调实验

下表补充了一些实验，用于回答有关各种设计选择的其他问题。第一行采用与本章主体内容相同的设置，作为参考。
例如，

- 比较第 1 行和第 2 行，可以回答：“训练最后一个 Token 还是第一个 Token，性能有什么差别？”
- 比较第 1 行和第 3 行，可以回答：“只训练最后一层而不是最后一个块，性能有什么差别？”
- 以此类推。

&nbsp;

|      | 模型               | 权重   | 可训练 Token 位置       | 可训练层         | 上下文长度                                             | 训练准确率 | 验证准确率 | 测试准确率 | 训练时间  | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | 预训练 | last                     | last_block       | 最长训练样本（120）                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | 预训练 | first                    | last_block       | 最长训练样本（120）                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | 预训练 | last                     | last_layer       | 最长训练样本（120）                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | 预训练 | last                     | last_two_blocks  | 最长训练样本（120）                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | 预训练 | last                     | all              | 最长训练样本（120）                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | 预训练 | last                     | last_block       | 最长训练样本（120）                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | 预训练 | last                     | last_block       | 最长训练样本（120）                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | 预训练 | last                     | last_block       | 最长训练样本（120）                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | 预训练 | last                     | all              | 最长训练样本（120）                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | 随机   | last                     | all              | 最长训练样本（120）                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | 预训练 | last                     | LoRA             | 最长训练样本（120）                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | 预训练 | last                     | LoRA             | 最长训练样本（120）                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 上下文长度（1024）                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 可变长度：不填充（批大小 1）                        | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 可变长度：不填充（批大小 8）                        | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 灵活（最后一个非填充位置）                           | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 最长训练样本（120），但没有因果掩码                   | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | 预训练 | last                     | last_block       | 最长训练样本（120），填充使用 `ignore_index`          | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | 预训练 | 最后位置 + 池化嵌入        | last_block       | 最长训练样本（120）                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### 用法

可以使用以下命令重现实验：

- 第 1 行：`python additional_experiments.py`
- 第 2 行：`python additional_experiments.py --trainable_token_pos first`
- 第 3 行：`python additional_experiments.py --trainable_layers last_layer`
- 第 4 行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第 5 行：`python additional_experiments.py --trainable_layers all`
- 第 6 行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第 7 行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第 8 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第 9 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)" --trainable_layers all`
- 第 10 行：`python additional_experiments.py --weights random --trainable_layers all`
- 第 11 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第 12 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第 13 行：`python additional_experiments.py --context_length "model_context_length"`
- 第 14 行：`python additional_experiments.py --no_padding --batch_size 1`
- 第 15 行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第 16 行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第 17 行：`python additional_experiments.py --disable_causal_mask`
- 第 18 行：`python additional_experiments.py --ignore_index 50256`
- 第 19 行：`python additional_experiments.py --average_embeddings`

我特意把大语言模型和数据集都设得很小，这样即使没有 GPU，你也可以在普通笔记本电脑（例如 MacBook Air M3）上用大约 15 分钟完成训练（采用默认设置）。

&nbsp;

### 结果解读

1. **训练最后一个还是第一个输出 Token 位置（第 1 行与第 2 行）**：训练最后一个输出 Token 位置的表现明显优于训练第一个位置。由于因果自注意力掩码的存在，这种改进符合预期。
2. **训练最后一个 Transformer 块还是最后一层（第 1 行与第 3 行）**：训练整个最后一个 Transformer 块的效果也明显优于只训练最后一层。
3. **训练最后一个还是最后两个 Transformer 块（第 1 行与第 4 行）**：与只训练最后一个块相比，训练最后两个 Transformer 块使准确率明显提高了 3.33%。
4. **训练最后一个 Transformer 块还是所有层（第 1 行与第 5 行）**：训练所有层比只训练最后一个 Transformer 块略有提升（约 2%），但训练时间几乎增加到三倍。此外，其表现不如只训练 12 个 Transformer 块中的最后两个。
5. **使用更大的预训练模型（第 1 行与第 6 行，以及第 1 行与第 7、8 行）**：使用大 3 倍的预训练模型会导致结果变差。不过，使用大 5 倍的模型会比初始模型表现更好，这与预期一致。同样，大 12 倍的模型会进一步提升预测表现。（medium 模型的预训练效果或许不佳，或者这套特定的微调配置对该模型效果较差。）
6. **使用随机权重模型还是预训练权重模型（第 1、5 行与第 10 行）**：使用随机权重模型得到的结果略差（分别低 3% 和 1.3%）。
7. **使用 LoRA（低秩适配）还是训练所有层（第 11 行与第 5 行，以及第 12 行与第 9 行）**：冻结模型并添加可训练的 LoRA 层（详情见[附录 E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)），是训练模型所有参数的一种可行替代方案，甚至能将表现提高 1 个百分点（第 11 行与第 5 行）。从使用 LoRA 时训练准确率与验证准确率之间的差距约低 1% 可以看出，这可能是过拟合减少所致。此外，由于需要更新的参数较少，使用 LoRA 也更节省内存。训练较大的模型时（第 12 行与第 9 行），我们还可以看到 LoRA 的训练速度更快（5.79 分钟，而非 8.12 分钟）。
8. **将输入填充到完整上下文长度还是最长训练样本长度（第 1 行与第 13 行）**：将输入填充到模型支持的完整上下文长度，表现明显更差。
9. **填充与不填充（第 1 行与第 14、15 行，以及第 16 行）**：`--no_padding` 选项会关闭数据集填充；由于输入长度不一，这要求使用批大小为 1 来训练模型。这样可以提高测试准确率，但训练时间更长。第 15 行还启用了 8 步梯度累积，以达到与其他实验相同的批大小；这有助于减少过拟合，并略微提高测试集准确率。第 16 行采用了填充，但根据最后一个非填充 Token 选择 Token 位置。从数学上说，第 16 行应与使用梯度累积的第 15 行相似。不过，在 Token 数量不相等的情况下，梯度累积会遇到一些问题，因此结果可能略有差异（[这篇博客文章](https://unsloth.ai/blog/gradient)对此有讨论）。
10. **禁用因果注意力掩码（第 1 行与第 17 行）**：禁用多头注意力模块中的因果注意力掩码。这意味着所有 Token 都可以关注其他所有 Token。与使用因果掩码的 GPT 模型相比，模型准确率略有提高。
11. **在损失计算和反向传播中忽略填充索引（第 1 行与第 18 行）**：设置 `--ignore_index 50256` 会在 PyTorch 的 `cross_entropy` 损失函数中排除 `<|endoftext|>` 填充 Token。在这个例子中，该设置没有效果，因为我们已替换输出层，使二元分类示例中的 Token ID 只能是 0 或 1。不过，在第 7 章对模型进行指令微调时，这个设置很有用。
12. **对所有 Token 的嵌入取平均值（第 1 行与第 19 行）**：设置 `--average_embeddings` 会对所有 Token 的嵌入取平均值。如果不使用此选项（默认行为），则只考虑指定 Token 位置（由 `--trainable_token_pos` 指定，例如最后一个 Token）的输出嵌入。启用 `--average_embeddings` 后，会对所有 Token 的嵌入取平均池化，并放入由 `--trainable_token_pos` 选定的位置（默认是最后一个 Token）。可以看到，这会将表现从 95.00% 提高到 96.33%，运行时间仅略有增加（从 0.28 分钟到 0.32 分钟）；实际使用时值得考虑。
