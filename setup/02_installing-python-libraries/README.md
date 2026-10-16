# 安装本书使用的 Python 包与库

本文进一步说明如何核对已安装的 Python 版本和包。（有关安装 Python 和 Python 包的更多说明，请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 目录。）

本书使用了[此处](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt)列出的库。更新版本很可能也兼容，但如果运行代码时遇到问题，可以退回使用这些库版本。



> **注意：**
> 如果按照[方案 1：使用 uv](../01_optional-python-setup-preferences/README.md) 使用 `uv`，可以将以下命令中的 `pip` 替换为 `uv pip`。例如，将 `pip install -r requirements.txt` 改为 `uv pip install -r requirements.txt`。



最方便的安装方式是使用本代码仓库根目录中的 `requirements.txt`，执行以下命令：

```bash
pip install -r requirements.txt
```

也可以通过 GitHub 地址安装：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


安装完成后，请通过以下命令检查是否已安装全部包，以及版本是否符合要求：

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

也建议在 JupyterLab 中运行本目录的 `python_environment_check.ipynb` 来检查版本，理想情况下，结果应与上面一致。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

如果出现以下问题，很可能是 JupyterLab 实例连接到了错误的 conda 环境：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

此时，可以使用 `watermark` 的 `--conda` 标志，检查 JupyterLab 实例是否在正确的 conda 环境中打开：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## 安装 PyTorch

与其他 Python 库或包一样，PyTorch 可以通过 pip 安装。例如：

```bash
pip install torch
```

不过，PyTorch 是一个同时提供 CPU 和 GPU 兼容代码的综合性库，安装时可能需要额外配置和说明，详见本书 *A.1.3 安装 PyTorch*。

同时强烈建议参考 PyTorch 官网 [https://pytorch.org](https://pytorch.org) 上的安装指南菜单。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



&nbsp;
## JupyterLab 使用建议

如果你通过 JupyterLab 而非 VSCode 查看 Notebook 代码，请注意：JupyterLab 的较新版本在默认设置下存在滚动问题。建议进入 Settings → Settings Editor（设置 → 设置编辑器），将“Windowing mode”设为“none”（如下图所示），这似乎能解决问题。


![Jupyter 显示问题 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 显示问题 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---




如有问题，欢迎在[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)交流。
