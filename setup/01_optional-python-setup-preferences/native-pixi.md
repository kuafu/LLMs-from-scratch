# 使用原生 pixi 命令管理 Python 和包

本教程面向偏好原生 `pixi` 命令而非 `conda`、`pip` 等传统环境和包管理工具的读者，是 [`./native-uv.md`](native-uv.md) 的替代方案。

如 [`./native-uv.md`](native-uv.md) 所述，pixi 在底层使用 `uv add`。

Pixi 和 uv 都是现代 Python 包与环境管理工具，但 pixi 与 conda 类似，是面向多种语言的包管理工具，除了 Python，还能管理其他语言；uv 则专注于 Python，针对极快的依赖解析和包安装进行了优化。

如果需要支持多种语言的包管理工具，或偏好类似 conda 的声明式环境管理方式，可以选择 pixi 而非 uv。更多说明请参阅 [pixi 官方文档](https://pixi.sh/latest/)。

本教程使用运行 macOS 的计算机演示；Linux 上的流程类似，也可能适用于其他操作系统。

&nbsp;
## 1. 安装 pixi

可以根据操作系统按以下方式安装 pixi。

<br>

**macOS 与 Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

或者

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

从[官方文档](https://pixi.sh/latest/installation/#__tabbed_1_2)下载安装程序，或运行其中列出的 PowerShell 命令。



> **注意：**
> 更多安装方式请参阅 [pixi 官方文档](https://pixi.sh/latest/)。


&nbsp;
## 2. 安装 Python

可以使用 pixi 安装 Python：

```bash
pixi add python=3.10
```

> **注意：**
> 为确保与 PyTorch 兼容，建议安装比最新发布版本至少早两个版本的 Python。例如，如果最新版为 Python 3.13，建议安装 3.10 或 3.11。可以访问 [python.org](https://www.python.org) 了解最新 Python 版本。

&nbsp;
## 3. 安装 Python 包与依赖

要安装 `pixi.toml` 文件（例如本 GitHub 仓库根目录中的文件）列出的全部必需包，请运行以下命令，假设该文件位于当前终端目录：

```bash
pixi install
```

> **注意：**
> 如果遇到依赖问题（例如在 Windows 上），可以随时退回使用 pip：`pixi run pip install -U -r requirements.txt`

默认情况下，`pixi install` 会创建一个专属于本项目的独立虚拟环境。

可以通过 `pixi add` 安装 `pixi.toml` 中没有列出的新包，例如：

```bash
pixi add packaging
```

也可以通过 `pixi remove` 移除包，例如：

```bash
pixi remove packaging
```

&nbsp;
## 4. 运行 Python 代码

现在你的环境应该已经可以运行本仓库的代码。

你还可以运行本仓库的 `python_environment_check.py` 脚本，检查环境：

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**启动 JupyterLab**

可以通过以下命令启动 JupyterLab 实例：

```bash
pixi run jupyter lab
```


---

如有问题，欢迎在[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)交流。
