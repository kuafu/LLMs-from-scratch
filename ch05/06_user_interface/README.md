# 构建与预训练 LLM 交互的用户界面



这个附加内容文件夹包含一个类似 ChatGPT 的用户界面代码，可用于与第 5 章中的预训练 LLM 交互，如下图所示。



![Chainlit 用户界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)



实现这个用户界面时，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖项

首先，通过以下命令安装 `chainlit` 包：

```bash
pip install chainlit
```

（或者，执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`app_orig.py`](app_orig.py)：此文件加载并使用 OpenAI 的原始 GPT-2 权重。
2. [`app_own.py`](app_own.py)：此文件加载并使用我们在第 5 章中生成的 GPT-2 权重。使用它之前，需要先运行 [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) 文件。

（打开并查看这些文件可以了解更多信息。）

在终端中运行以下命令之一来启动 UI 服务器：

```bash
chainlit run app_orig.py
```

or

```bash
chainlit run app_own.py
```

运行上述任一命令后，应该会打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令并将本地地址复制到浏览器地址栏中（通常地址为 `http://localhost:8000`）。
