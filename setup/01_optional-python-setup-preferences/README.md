# Python 环境配置建议



安装 Python 和配置计算环境有多种方法。这里分享我个人偏好的做法。

<br>

> **注意：**
> 如果你在 Google Colab 上运行 Notebook，并希望安装依赖，只需在 Notebook 顶部新建单元格，运行以下代码，然后跳过本教程的其余内容：
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

下面各节介绍如何在本地计算机上管理 Python 环境和包。

我长期使用 [Conda](https://anaconda.org/anaconda/conda) 和 [pip](https://pypi.org/project/pip/)。不过，近来 [uv](https://github.com/astral-sh/uv) 越来越受欢迎，因为它能更快速、高效地安装包并解析依赖。

建议从“方案 1：使用 uv”开始，这是在 2025 年更现代的做法。如果“方案 1”遇到问题，可以考虑“方案 2：使用 Conda”。

本教程使用运行 macOS 的计算机演示；Linux 上的流程类似，也可能适用于其他操作系统。


&nbsp;
# 方案 1：使用 uv

本节介绍通过 `uv` 的 `uv pip` 接口配置 Python 和安装包的流程。对于用过 pip 的 Python 用户，`uv pip` 接口可能比原生 `uv` 命令更熟悉。

&nbsp;
> **注意：**
> 安装 Python 和使用 `uv` 还有其他方式。例如，可以直接通过 `uv` 安装 Python，再使用 `uv add` 代替 `uv pip install`，进一步加快包管理。
>
> 如果你使用 macOS 或 Linux，并偏好原生 `uv` 命令，请参阅 [./native-uv.md 教程](./native-uv.md)。同时也建议阅读 [`uv` 官方文档](https://docs.astral.sh/uv/)。
>
> `uv add` 语法同样适用于 Windows。不过，我发现 `pyproject.toml` 中的部分依赖会在 Windows 上引发问题。因此，对于 Windows 用户，我推荐改用 `pixi`，其 `pixi add` 工作流程与 `uv add` 类似。更多说明见 [./native-pixi.md 教程](./native-pixi.md)。
>
> 虽然 `uv add` 和 `pixi add` 在速度上更有优势，但我认为 `uv pip` 稍易上手，适合初学者。不过，如果你刚开始接触 Python 包管理，原生 `uv` 接口也是从头学习的好机会。这也是我目前使用 `uv` 的方式，但对于习惯 `pip` 和 `conda` 的用户，入门门槛可能略高。




&nbsp;
## 1. 安装 Python（如尚未安装）

如果之前没有手动在系统上安装过 Python，我强烈建议先安装。这有助于避免与操作系统内置 Python 发生潜在冲突，从而引发问题。

即使之前安装过 Python，也请在终端执行以下代码，确认版本是否较新（建议 3.10 或更新版本）：

```bash
python --version
```
如果返回 3.10 或更新版本，则无需进一步操作。

&nbsp;
> **注意：**
> 如果 `python --version` 提示没有安装 Python，也请检查 `python3 --version`，因为系统可能配置为使用 `python3` 命令。

&nbsp;
> **注意：**
> 为确保与 PyTorch 兼容，建议安装比最新发布版本至少早两个版本的 Python。例如，如果最新版为 Python 3.13，建议安装 3.10 或 3.11。

如果尚未安装 Python，或版本较旧，可以根据操作系统按以下方法安装。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="未找到 Python">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

如果使用 Homebrew，可以通过以下命令安装 Python：

```bash
brew install python@3.10
```

也可以从官网下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python 版本">

<br>

**Windows**

从官网下载并运行安装程序：[https://www.python.org/downloads/](https://www.python.org/downloads/)。


&nbsp;

## 2. 创建虚拟环境

强烈建议在独立的虚拟环境中安装 Python 包，以免修改操作系统可能依赖的系统级包。要在当前目录创建虚拟环境，请执行以下三个步骤。

<br>

**1. 安装 uv**

```bash
pip install uv
```

<br>

**2. 创建虚拟环境**

```bash
uv venv --python=python3.10
```

<br>

**3. 激活虚拟环境**

```bash
source .venv/bin/activate
```

&nbsp;
> **注意：**
> 如果使用 Windows，可能需要将上面的命令替换为 `source .venv/Scripts/activate` 或 `.venv/Scripts/activate`。



每次启动新的终端会话，都需要激活虚拟环境。例如，重启终端或计算机后，第二天要继续处理项目时，只需在项目目录运行 `source .venv/bin/activate`，即可重新激活虚拟环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="虚拟环境已激活">

需要时，可以执行 `deactivate` 命令停用环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="虚拟环境已停用">

&nbsp;
## 3. 安装包

激活虚拟环境后，可以使用 `uv` 安装 Python 包。例如：

```bash
uv pip install packaging
```

要安装 `requirements.txt` 文件（例如本 GitHub 仓库根目录中的文件）列出的全部必需包，请运行以下命令，假设该文件位于当前终端目录：

```bash
uv pip install -r requirements.txt
```


也可以直接从仓库安装最新依赖：

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="uv 安装">

&nbsp;

> **注意：**
> 如果上述命令因某些依赖而出现问题（例如在 Windows 上），可以随时退回使用普通 pip：
> `pip install -r requirements.txt`
> 或者
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

&nbsp;

> **补充资料的可选依赖：**
> 要安装各项补充资料使用的可选依赖，请在项目根目录安装 `bonus` 依赖组：
>  `uv pip install --group bonus`
> 如果你不想在之后学习可选补充资料时分别安装依赖，这种方式会更方便。

<br>

**完成配置**

配置完成！现在你的环境应该已经可以运行本仓库的代码。

你还可以运行本仓库的 `python_environment_check.py` 脚本，检查环境：

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="环境检查">

如果某个包出现问题，可以尝试用以下命令重新安装：

```bash
uv pip install packagename
```

（这里的 `packagename` 是占位名称，需要替换为出问题的包名。）

如果问题仍然存在，可以在 GitHub 上[发起讨论](https://github.com/rasbt/LLMs-from-scratch/discussions)，或尝试下方“方案 2：使用 Conda”中的步骤。

<br>

**开始使用代码**

完成全部配置后，就可以使用代码文件了。例如，运行以下命令启动 [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)：

```bash
jupyter lab
```

&nbsp;
> **注意：**
> 如果 jupyter lab 命令遇到问题，也可以通过虚拟环境内的完整路径启动。例如，Linux/macOS 上使用 `.venv/bin/jupyter lab`，Windows 上使用 `.venv\Scripts\jupyter-lab`。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="uv 安装">

&nbsp;
<br>
<br>
&nbsp;

# 方案 2：使用 Conda



本节介绍通过 [miniforge](https://github.com/conda-forge/miniforge) 使用 [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8) 配置 Python 和安装包的流程。

本教程使用运行 macOS 的计算机演示；Linux 上的流程类似，也可能适用于其他操作系统。


&nbsp;
## 1. 下载并安装 Miniforge

从 GitHub 仓库[下载 miniforge](https://github.com/conda-forge/miniforge)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="下载" width="600px">

根据操作系统，下载的文件应为 `.sh`（macOS、Linux）或 `.exe`（Windows）。

对于 `.sh` 文件，请打开命令行终端，执行以下命令：

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中，`Desktop/` 是下载 Miniforge 安装程序的目录。在你的计算机上，可能需要改成 `Downloads/`。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="安装 Miniforge" width="600px">

接下来按提示逐步操作，使用 Enter 键确认。


&nbsp;
## 2. 创建新的虚拟环境

安装成功后，建议创建一个名为 `LLMs` 的新虚拟环境，执行：

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="新建环境" width="600px">

> 很多科学计算库不会立即支持最新版本的 Python。因此，安装 PyTorch 时，建议使用比最新版早一到两个版本的 Python。例如，如果最新版本为 Python 3.13，建议使用 Python 3.10 或 3.11。

接下来激活新建的虚拟环境（每次打开新的终端窗口或标签页时，都需要执行此步骤）：

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="激活环境" width="600px">


&nbsp;
## 可选：美化终端

如果你想让终端像我的一样显示当前激活的虚拟环境，可以了解 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

&nbsp;
## 3. 安装新的 Python 库



现在可以使用 `conda` 包安装工具安装新的 Python 库。例如，可以按以下方式安装 [JupyterLab](https://jupyter.org/install) 和 [watermark](https://github.com/rasbt/watermark)：

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="使用 conda 安装" width="600px">



你也仍然可以用 `pip` 安装库。默认情况下，`pip` 应当关联到新建的 `LLMs` conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="检查 pip" width="600px">

&nbsp;
## 4. 安装 PyTorch

与其他 Python 库或包一样，PyTorch 可以通过 pip 安装。例如：

```bash
pip install torch
```

不过，PyTorch 是一个同时提供 CPU 和 GPU 兼容代码的综合性库，安装时可能需要额外配置和说明，详见本书 *A.1.3 安装 PyTorch*。

同时强烈建议参考 PyTorch 官网 [https://pytorch.org](https://pytorch.org) 上的安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

&nbsp;
## 5. 安装本书使用的 Python 包与库

有关安装所需库的说明，请参阅[安装本书使用的 Python 包与库](../02_installing-python-libraries/README.md)。

<br>

---




如有问题，欢迎在[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)交流。
