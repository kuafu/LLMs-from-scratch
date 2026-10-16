# 可选环境配置说明


本文介绍配置计算机并运行本仓库代码的不同方法。建议从上到下浏览各节，再选择最适合自己需求的方法。

&nbsp;

## 快速开始

如果计算机上已经安装 Python，最快的入门方式是在本代码仓库的根目录运行以下 pip 命令，安装 [../requirements.txt](../requirements.txt) 中列出的依赖：

```bash
pip install -r requirements.txt
```

<br>

> **注意：** 如果你在 Google Colab 上运行 Notebook，并希望安装依赖，只需在 Notebook 顶部新建一个单元格，运行以下代码：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> 另外，克隆仓库后，可以在项目根目录执行 `uv pip install --group bonus`，安装所有补充资料的依赖。如果你不想在日后学习各项可选补充资料时分别安装依赖，这种方式会更方便。



在下面的视频中，我分享了自己在计算机上配置 Python 环境的方法：

<br>
<br>

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# 本地环境配置

本节介绍在本地运行本书代码的建议。本书主要章节的代码可以在普通笔记本电脑上于合理时间内运行，无需专用硬件。我已在 M3 MacBook Air 笔记本电脑上测试过所有主要章节。此外，如果你的笔记本或台式计算机配有 NVIDIA GPU，代码会自动使用它。

&nbsp;
## 配置 Python

如果计算机尚未配置 Python，可以参考以下目录，其中介绍了我个人偏好的 Python 配置方法：

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下方的“使用 DevContainer”一节提供了另一种在计算机上安装项目依赖的方法。

&nbsp;

## 使用 Docker DevContainer

除了上方“配置 Python”一节的方法，如果你希望隔离项目依赖和配置，Docker 也是一种有效的开发环境方案。它无需手动安装软件包和库，并能确保开发环境一致。关于配置 Docker 和使用 DevContainer 的更多说明，请参阅：

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code 编辑器

代码编辑器有许多不错的选择。我偏好流行的开源编辑器 [Visual Studio Code（VSCode）](https://code.visualstudio.com)，它可以方便地通过实用插件和扩展增强功能，详见下方“VSCode 扩展”一节。macOS、Linux 和 Windows 的下载说明见 [VSCode 官网](https://code.visualstudio.com)。

&nbsp;

## VSCode 扩展

如果你使用 Visual Studio Code（VSCode）作为主要代码编辑器，可以在 `.vscode` 子目录中找到推荐扩展。这些扩展提供了对本仓库有帮助的增强功能和工具。

要安装它们，请在 VSCode 中打开此 setup 文件夹（File → Open Folder…，即“文件 → 打开文件夹…”），然后点击右下角弹出菜单中的“Install（安装）”按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

你也可以将 `.vscode` 扩展配置目录移到本 GitHub 仓库的根目录：

```bash
mv setup/.vscode ./
```

这样，每次打开 `LLMs-from-scratch` 主目录时，VSCode 都会自动检查系统是否已经安装推荐扩展。

&nbsp;

# 云端资源

本节介绍在云端运行本书代码的替代方案。

虽然代码可以在没有独立 GPU 的普通笔记本或台式计算机上运行，但配备 NVIDIA GPU 的云平台能明显缩短运行时间，尤其是第 5 至第 7 章的代码。

&nbsp;

## 使用 Lightning Studio

为获得顺畅的云端开发体验，我推荐 [Lightning AI Studio](https://lightning.ai/) 平台。它支持创建持久化环境，并在云端 CPU 和 GPU 上使用 VSCode 和 Jupyter Lab。

新建 Studio 后，可以打开终端，执行以下步骤来克隆仓库并安装依赖：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

（与 Google Colab 不同，Lightning AI Studio 环境会持久保存，即使在 CPU 与 GPU 机器之间切换也是如此，因此这些步骤只需执行一次。）

接下来，找到要运行的 Python 脚本或 Jupyter Notebook。你也可以方便地接入 GPU，加快代码运行，例如在第 5 章预训练 LLM，或在第 6、7 章微调模型时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用 Google Colab

要使用云端 Google Colab 环境，请访问 [https://colab.research.google.com/](https://colab.research.google.com/)，从 GitHub 菜单中打开相应章节的 Notebook，或如下图所示，将 Notebook 拖入 *Upload（上传）* 区域。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


同时，请确保将相关文件（数据集文件及 Notebook 导入的 .py 文件）也上传到 Colab 环境，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


你还可以按下图所示修改 *Runtime（运行时）*，选择在 GPU 上运行代码。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;

# 遇到问题？

如有问题，欢迎通过本 GitHub 仓库的[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)交流。
