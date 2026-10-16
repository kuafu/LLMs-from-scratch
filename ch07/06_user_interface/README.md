# 构建用于交互式使用指令微调 GPT 模型的用户界面



此附加目录包含用于运行类似 ChatGPT 的用户界面的代码，可与第 7 章中的指令微调 GPT 交互，如下所示。



![Chainlit 用户界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)



我们使用开源的 [Chainlit Python 软件包](https://github.com/Chainlit/chainlit)来实现这个用户界面。

&nbsp;
## 第 1 步：安装依赖项

首先，通过以下命令安装 `chainlit` 软件包：

```bash
pip install chainlit
```

（或者，运行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 第 2 步：运行 `app` 代码

[`app.py`](app.py) 文件包含用户界面代码。打开并查看该文件可以了解更多信息。

此文件会加载并使用我们在第 7 章中生成的 GPT-2 权重。为此，必须先执行 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb)。

在终端中运行以下命令以启动用户界面服务器：

```bash
chainlit run app.py
```

运行上述命令后，浏览器通常会打开一个新标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令，并将本地地址复制到浏览器地址栏中（通常地址为 `http://localhost:8000`）。
