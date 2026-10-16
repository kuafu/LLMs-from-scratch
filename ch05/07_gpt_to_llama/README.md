# 将 GPT 转换为 Llama



本文件夹包含将第 4 章和第 5 章中的 GPT 实现转换为 Meta AI 的 Llama 架构的代码，建议按以下顺序阅读：

- [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb)：包含逐步将 GPT 转换为 Llama 2 7B 并加载 Meta AI 预训练权重的代码
- [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb)：包含将 Llama 2 模型转换为 Llama 3、Llama 3.1 和 Llama 3.2 的代码
- [standalone-llama32.ipynb](standalone-llama32.ipynb)：独立实现 Llama 3.2 的笔记本

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">


&nbsp;
### 通过 `llms-from-scratch` 软件包使用 Llama 3.2

如果想更方便地使用 Llama 3.2 1B 和 3B 模型，也可以使用基于本仓库源代码的 `llms-from-scratch` PyPI 软件包：[pkg/llms_from_scratch](../../pkg/llms_from_scratch)。

&nbsp;
#### 1）安装

```bash
pip install llms_from_scratch blobfile
```

（请注意，加载分词器需要 `blobfile`。）

&nbsp;
#### 2）模型和文本生成设置

指定要使用的模型：

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

用户可以自行设置以下基本文本生成参数。请注意，在文本生成示例中，建议使用的 8192 词元上下文长度需要约 3 GB 显存。

```python
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3）下载并加载权重

下面的代码会根据上面选择的模型自动下载权重文件：

```python
import os
import requests

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {MODEL_FILE}")
```

然后按如下方式加载模型权重：

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4）初始化分词器

以下代码会下载并初始化分词器：

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
#### 5）生成文本

最后，我们可以通过以下代码生成文本：

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

使用 Llama 3.2 1B Instruct 模型时，输出应与下方示例相似：

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
#### 专业提示 1：使用 FlashAttention 加快推理

你可以使用 `Llama3ModelFast` 作为 `Llama3Model` 的直接替代品。更多信息请查看 [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py) 代码。

`Llama3ModelFast` 将 `GroupedQueryAttention` 模块中我从零实现的缩放点积代码替换为 PyTorch 的 `scaled_dot_product` 函数；该函数会在 Ampere 或更新架构的 GPU 上使用 `FlashAttention`。

下表展示了在 A100 上的性能对比：

|                 | 令牌/秒    | 内存    |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
<a id="pro-tip-2-speed-up-inference-with-compilation"></a>
#### 专业提示 2：通过编译加快推理


如需获得最高 4× 的加速，可将

```python
model.to(device)
```

替换为

```python
model = torch.compile(model)
model.to(device)
```

注意：编译前期会产生数分钟的显著开销，并且要在第一次调用 `generate` 后才会体现加速效果。

下表展示了在 A100 上连续调用 `generate` 时的性能对比：

|                 | 令牌/秒    | 内存    |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
<a id="pro-tip-3-speed-up-inference-with-compilation"></a>
#### 专业提示 3：通过编译加快推理

在 CPU 上运行模型时，使用带 KV 缓存 的 `Llama3Model` 直接替代品可以显著提升推理性能。（有关 KV 缓存 的更多信息，请参阅我的文章[从零理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。）

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

请注意，由于更容易计算，表中只列出了 Nvidia CUDA 设备的峰值内存用量。不过，其他设备使用相似的精度格式，因此其内存用量可能也相近；在此生成 150 词元文本时，KV 缓存 的存储甚至会进一步降低内存用量。（不过，不同设备可能以不同方式实现矩阵乘法，因而峰值内存需求可能不同；上下文长度增加时，KV 缓存 的内存占用也可能大幅增长。）

| 模型        | 模式              | 硬件            | 令牌/秒    | GPU 内存（VRAM）   |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | 普通              | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | 普通（已编译）    | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV 缓存          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV 缓存（已编译）| Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | 普通              | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | 普通（已编译）    | Mac Mini M4 GPU | 错误       | -                 |
| Llama3Model | KV 缓存          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV 缓存（已编译）| Mac Mini M4 GPU | 错误       | -                 |
|             |                   |                 |            |                   |
| Llama3Model | 普通              | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | 普通（已编译）    | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV 缓存          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV 缓存（已编译）| Nvidia A100 GPU | 161        | 3.61 GB           |

请注意，以上所有设置都经过测试，生成的文本输出相同。
