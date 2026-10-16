# 使用原生 uv 命令管理 Python 和包

本教程面向偏好原生 `uv` 命令而非 `uv pip` 接口的读者，是 [README.md](./README.md) 中“方案 1：使用 uv”的替代方案。`uv pip` 比普通 `pip` 更快，而原生 `uv` 接口的开销更低，也无需处理 PyPy 包依赖管理的旧版兼容支持，因此比 `uv pip` 还快。

下表对比不同依赖和包管理方式的速度。这里比较的是安装时解析包依赖的速度，而非已安装包的运行性能。对于本项目，安装包只需进行一次，因此可以根据整体便利性选择方案，不必仅看安装速度。


| 命令 | 速度对比 |
|-----------------------|-----------------|
| `conda install <pkg>` | 最慢（基准） |
| `pip install <pkg>` | 比上一种方式快 2-10 倍 |
| `uv pip install <pkg>` | 比上一种方式快 5-10 倍 |
| `uv add <pkg>` | 比上一种方式快 2-5 倍 |

本教程主要介绍 `uv add`。


其余内容与 [README.md](./README.md) 中的“方案 1：使用 uv”类似，介绍如何使用 `uv` 配置 Python 和安装包。

本教程使用运行 macOS 的计算机演示；Linux 上的流程类似，也可能适用于其他操作系统。


&nbsp;
## 1. 安装 uv

可以根据操作系统按以下方式安装 uv。

<br>

**macOS 与 Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

&nbsp;

> **注意：**
> 更多安装方式请参阅 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)。

&nbsp;
## 2. 安装 Python 包与依赖

要安装 `pyproject.toml` 文件（例如本 GitHub 仓库根目录中的文件）列出的全部必需包，请运行以下命令，假设该文件位于当前终端目录：

```bash
uv sync --dev --python 3.11
```

> **注意：**
> 如果系统上没有 Python 3.11，uv 会为你下载并安装。
> 为确保与 PyTorch 兼容，建议使用比最新发布版本早 1-3 个版本的 Python。例如，如果最新版为 Python 3.13，建议使用 3.10、3.11 或 3.12。可以访问 [python.org](https://www.python.org/downloads/) 了解最新 Python 版本。

> **注意：**
> 如果上述命令因某些依赖而出现问题（例如在 Windows 上），可以随时退回使用普通 pip：
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`


上面的 `uv sync` 命令会在 `.venv` 子目录中创建独立虚拟环境。（如果想删除虚拟环境并重新开始，只需删除 `.venv` 目录。）

可以通过 `uv add` 安装 `pyproject.toml` 中没有列出的新包，例如：

```bash
uv add packaging
```

也可以通过 `uv remove` 移除包，例如：

```bash
uv remove packaging
```



&nbsp;
## 3. 运行 Python 代码

<br>

现在你的环境应该已经可以运行本仓库的代码。

你还可以运行本仓库的 `python_environment_check.py` 脚本，检查环境：

```bash
uv run python setup/02_installing-python-libraries/python_environment_check.py
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-run-check.png?1" width="700" height="auto" alt="uv 安装">


<br>

**启动 JupyterLab**

可以通过以下命令启动 JupyterLab 实例：

```bash
uv run jupyter lab
```

**省去 `uv run` 命令**

如果觉得每次输入 `uv run` 很麻烦，可以按以下方式手动激活虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows（PowerShell）上：

```bash
.venv\Scripts\activate
```

然后可以通过以下命令运行脚本：

```bash
python script.py
```

通过以下命令启动 JupyterLab：

```bash
jupyter lab
```

&nbsp;
> **注意：**
> 如果 jupyter lab 命令遇到问题，也可以通过虚拟环境内的完整路径启动。例如，Linux/macOS 上使用 `.venv/bin/jupyter lab`，Windows 上使用 `.venv\Scripts\jupyter-lab`。

&nbsp;


&nbsp;

## 可选：手动管理虚拟环境

你也仍然可以使用 `uv pip install` 直接从仓库安装依赖。但它不会像 `uv add` 一样将依赖记录在 `uv.lock` 文件中，而且需要手动创建并激活虚拟环境：

<br>

**1. 创建新的虚拟环境**

运行以下命令手动创建虚拟环境，它会保存在新建的 `.venv` 子目录中：

```bash
uv venv --python=python3.10
```

<br>

**2. 激活虚拟环境**

接下来，需要激活新建的虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows（PowerShell）上：

```bash
.venv\Scripts\activate
```

<br>

**3. 安装依赖**

最后，通过 `uv pip` 接口从远程地址安装依赖：

```bash
uv pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```



---

如有问题，欢迎在[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)交流。
