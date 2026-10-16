# 使用 Project Gutenberg 数据集预训练 GPT

此目录中的代码用于在 Project Gutenberg 提供的免费图书上训练小型 GPT 模型。

正如 Project Gutenberg 网站所述：“绝大多数 Project Gutenberg 电子书在美国属于公共领域。”

有关如何使用 Project Gutenberg 提供的资源，请阅读 [Project Gutenberg 权限、许可及其他常见问题](https://www.gutenberg.org/policy/permission.html)页面。

&nbsp;
## 如何使用此代码

&nbsp;

### 1）下载数据集

本节中，我们使用 GitHub 仓库 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 中的代码下载 Project Gutenberg 图书。

截至本文撰写时，这需要约 50 GB 磁盘空间，耗时约 10–15 小时；如果 Project Gutenberg 此后增长较多，所需时间可能更长。

&nbsp;
#### Linux 和 macOS 用户的下载说明


Linux 和 macOS 用户可以按照以下步骤下载数据集（Windows 用户请参阅下面的说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设为工作目录，以便在此文件夹中克隆 `gutenberg` 仓库（运行提供的 `prepare_dataset.py` 和 `pretraining_simple.py` 脚本需要这样做）。例如，在 `LLMs-from-scratch` 仓库目录下，通过以下命令进入 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在该目录中克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 仓库目录：
```bash
cd gutenberg
```

4. 在 `gutenberg` 仓库目录中安装 *requirements.txt* 里列出的必需软件包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回 `03_bonus_pretraining_on_gutenberg` 文件夹
```bash
cd ..
```

&nbsp;
#### Windows 用户的特别说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码兼容 Linux 和 macOS。不过，Windows 用户需要做一些小调整，例如在 `subprocess` 调用中添加 `shell=True`，并替换 `rsync`。

另一种更简单的方式是在 Windows 上使用“适用于 Linux 的 Windows 子系统”（WSL）功能，通过 Ubuntu 运行 Linux 环境。更多信息请阅读 [Microsoft 官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install)和[教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保已安装 Python 3（可通过 `python3 --version` 检查；例如，对于 Python 3.10，可以使用 `sudo apt-get install -y python3.10` 安装），并在其中安装以下软件包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 有关设置 Python 和安装软件包的说明，请参阅[可选的 Python 设置偏好](../../setup/01_optional-python-setup-preferences/README.md)和[安装 Python 库](../../setup/02_installing-python-libraries/README.md)。
>
> 本仓库还提供了一个运行 Ubuntu 的 Docker 镜像。有关如何使用此镜像运行容器的说明，请参阅[可选的 Docker 环境](../../setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2）准备数据集

接下来，运行 `prepare_dataset.py` 脚本。它会将（截至本文撰写时的）60,173 个文本文件合并为数量更少、体积更大的文件，以便更高效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **Tip:**
> 请注意，为了保持简单，生成的文件以纯文本格式存储，且没有预先进行词元化。不过，如果你计划更频繁地使用此数据集或训练多个轮次，可以修改代码，以词元化形式存储数据集，从而节省计算时间。更多信息请参阅本页底部的*设计决策与改进*。

> **Tip:**
> 你可以选择更小的文件大小，例如 50 MB。这样会产生更多文件，但如果只是为了测试而在少量文件上快速预训练，这可能会很有用。


&nbsp;
### 3）运行预训练脚本

可以按如下方式运行预训练脚本。为便于说明，下面列出的附加命令行参数使用了默认值：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出格式如下：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **Tip:**
> 实际使用时，如果你使用 macOS 或 Linux，我建议使用 `tee` 命令，在终端打印日志的同时将日志输出保存到 `log.txt` 文件中：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **Warning:**
> 请注意，在 `gutenberg_preprocessed` 文件夹中一个约 500 MB 的文本文件上训练，使用 V100 GPU 大约需要 4 小时。
> 该文件夹包含 47 个文件，全部训练完成大约需要 200 小时（超过 1 周）。你可以考虑只在较少的文件上运行训练。


&nbsp;
## 设计决策与改进

请注意，出于教学目的，此代码力求简单精简。可以通过以下方式改进代码，以提升建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本，去掉每本书文件中的 Gutenberg 模板文本。
2. 更新数据准备和加载工具，预先对数据集进行词元化并以词元形式保存，这样每次调用预训练脚本时就不必重新词元化。
3. 更新 `train_model_simple` 脚本，加入[附录 D：为训练循环增添实用功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中介绍的功能，即余弦衰减、线性预热和梯度裁剪。
4. 更新预训练脚本以保存优化器状态（参见第 5 章的 *5.4 在 PyTorch 中加载和保存权重*；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加选项，以便在训练中断时加载已有的模型和优化器检查点并继续训练。
5. 添加更高级的日志记录工具（例如 Weights and Biases），以便实时查看损失曲线和验证曲线。
6. 添加分布式数据并行（DDP），并在多个 GPU 上训练模型（参见附录 A 的 *A.9.3 使用多个 GPU 训练*；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中的从零实现 `MultiheadAttention` 类替换为附加章节[高效的多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)中的高效 `MHAPyTorchScaledDotProduct` 类，该类通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或 [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型，以加快训练速度。
9. 实现梯度低秩投影（GaLore），进一步加快预训练过程。只需将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore)提供的 `GaLoreAdamW` 即可实现。
