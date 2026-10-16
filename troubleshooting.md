# 故障排查指南

本页汇总学习本书时可能遇到的常见问题和环境配置建议。

&nbsp;
## Notebook 图片加载问题

各章 Notebook 使用 Markdown 图片链接，图片托管于 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...`。这样可以控制仓库的下载大小，但图片能否显示也取决于图片托管网站和你的网络连接。

如果 `.ipynb` Notebook 中的图片无法显示：

- 在浏览器中直接打开一个图片地址，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果浏览器也无法加载该地址，原因很可能是网站、DNS、VPN、代理、防火墙或本地网络的临时问题，而非 Notebook 本身。
- 建议换一台设备或换一个网络再次检查该地址，例如用手机打开图片。如果手机能正常加载，很可能是计算机上的 VPN 或防火墙出了问题。
- 如果手机也无法加载图片，欢迎提交 GitHub [问题报告](https://github.com/rasbt/LLMs-from-scratch/issues)，帮助我进一步排查。

&nbsp;
## 更新仓库时保留个人 Notebook 修改

如果你希望修改 Notebook，同时继续接收仓库更新，请先复刻（Fork）仓库，再克隆自己的复刻仓库。书中主要 Notebook 与纸质书保持同步，除关键修复外通常不会修改；仓库更新主要用于添加补充资料。

Notebook 文件采用 JSON 格式，因此 Git 差异和合并冲突可能不易阅读。为避免不必要的冲突，建议将自己的实验与纳入版本控制的书中 Notebook 分开：

- 修改前先复制 Notebook，例如将 `ch02.ipynb` 复制为 `ch02_experiments.ipynb`。
- 将实验用 Notebook 放在单独的目录或自己的分支上。
- 通过名为 `upstream` 的远程仓库获取原仓库更新，仅在需要时才合并或变基。

创建并克隆复刻仓库：

1. 打开 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 点击 GitHub 页面右上角的 **Fork** 按钮。
3. 克隆自己的复刻仓库，将 `YOUR-USERNAME` 替换为你的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

然后将原仓库添加为 `upstream`，以便获取后续更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果确实需要合并修改过的 Notebook，可以考虑安装 [`nbdime`](https://nbdime.readthedocs.io/)，使用专门针对 Notebook 的差异比较与合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

更多背景信息请参阅 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

&nbsp;
## Apple Silicon 与 MPS 支持

部分 Notebook 和脚本会在 `cuda` 可用时使用它，否则回退到 `cpu`，而不选择 Apple 的 `mps` 后端。很多地方是有意不支持 `mps` 的，因为较早的 PyTorch/MPS 版本在若干示例中会产生不稳定或不一致的结果，尤其是在训练和微调时。

如果你使用 Apple Silicon Mac，遇到损失发散、损失突然大幅上升、生成文本质量差或结果与书中不一致等情况，请先在 `cpu` 上重新运行示例。如果希望加快训练并保持与书中一致的行为，建议在本地 NVIDIA GPU 或云端 GPU 上使用 `cuda`。

更新的 PyTorch 版本可能改善 MPS 的表现；只要仔细验证结果，你也可以在本地尝试 `mps`。不过，如果自行给脚本添加 `mps` 支持，请注意：`pin_memory=True`、`torch.compile` 等与 CUDA 相关的选项，以及 DDP／多 GPU 代码，可能需要单独添加条件判断。

更多背景信息请参阅 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

&nbsp;
## 其他问题

如有其他问题，欢迎提交新的 GitHub [问题报告](https://github.com/rasbt/LLMs-from-scratch/issues)。
