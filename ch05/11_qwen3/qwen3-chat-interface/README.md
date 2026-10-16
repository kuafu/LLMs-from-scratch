# 从零实现 Qwen3 并构建聊天界面



这个附加内容文件夹包含一个类似 ChatGPT 的用户界面代码，可用于与预训练 Qwen3 模型交互。



![Chainlit 用户界面示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



实现这个用户界面时，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖项

首先，通过以下命令安装 `chainlit` 包以及 [requirements-extra.txt](requirements-extra.txt) 列出的依赖项：

```bash
pip install -r requirements-extra.txt
```

或者，如果你使用 `uv`，请运行：

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## 步骤 2：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：此文件加载并使用思考模式下的 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面相同，但配置为记住消息历史。

（打开并查看这些文件可以了解更多信息。）

在终端中运行以下命令之一来启动 UI 服务器：

```bash
chainlit run qwen3-chat-interface.py
```

或者，如果你使用 `uv`，请运行：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述任一命令后，应该会打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令并将本地地址复制到浏览器地址栏中（通常地址为 `http://localhost:8000`）。
