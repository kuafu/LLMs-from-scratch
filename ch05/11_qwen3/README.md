# 从零实现 Qwen3

本文件夹中的 Jupyter 笔记本 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) 从零实现了 Qwen3 0.6B、1.7B、4B、8B 和 32B。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


本文件夹中的 Jupyter 笔记本 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) 从零实现了 30B-A3B 混合专家（MoE）模型，包括 Thinking、Instruct 和 Coder 模型变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Qwen3 从零实现代码

本文件夹中的独立笔记本按线性流程展示了从零实现的代码：

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb)：不含额外功能的 Qwen3 稠密模型
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb)：与上面相同，但加入了 KV 缓存 以提高推理效率
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)：类似第一个笔记本，但实现的是混合专家（MoE）变体
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb)：与上面相同，但加入了 KV 缓存 以提高推理效率

此外，我还将代码整理成了一个 Python 软件包[，见此处](../../pkg/llms_from_scratch/)（包含单元测试和 CI）；你可以按照下文说明运行它。

&nbsp;
# 训练

`Qwen3Model` 类的实现风格与 `GPTModel` 类类似，因此可以直接替代后者，用于第 5 章中的训练以及第 6、7 章中的微调。


&nbsp;
# 通过 `llms-from-scratch` 软件包使用 Qwen3

如果想更方便地使用 Qwen3 的从零实现，也可以使用基于本仓库源代码的 `llms-from-scratch` PyPI 软件包：[pkg/llms_from_scratch](../../pkg/llms_from_scratch)。

&nbsp;
#### 1）安装

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2）模型和文本生成设置

指定要使用的模型：

```python
USE_REASONING_MODEL = True
# Uses the base model if USE_REASONING_MODEL = False

USE_INSTRUCT_MODEL = False
# Uses the instruct mode (without reasoning) if 
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# This setting does have no effect if USE_REASONING_MODEL = False


# Use
# USE_REASONING_MODEL = True
# For Qwen3 Coder Flash model as well
```

用户可以自行设置以下基本文本生成参数。生成 150 个词元时，0.6B 模型需要约 1.5 GB 内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a）下载并加载 0.6B 模型的权重

以下代码会根据上面选择的模型（reasoning 或 base）自动下载权重文件。请注意，本节聚焦于 0.6B 模型。如果想使用更大的模型（1.7B、4B、8B 或 32B），请跳过本节并继续阅读 3b）节。

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后按如下方式加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b）下载并加载更大型 Qwen 模型的权重

如果想使用更大型的 Qwen 模型（例如 1.7B、4B、8B 或 32B），请使用以下代码，而不是 3a）节中的代码；此处需要额外的代码依赖项：

```bash
pip install safetensors huggingface_hub
```

然后使用以下代码（适当修改 `USE_MODEL`，以选择所需的模型大小）：

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，下载权重并将其加载到 `model` 中：

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # only required for the MoE models
del weights_dict  # delete weight dictionary to free up disk space
```


&nbsp;

#### 4）初始化分词器

以下代码会下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



&nbsp;

#### 5）生成文本

最后，我们可以通过以下代码生成文本：

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

使用 Qwen3 0.6B 推理模型时，输出应与下方示例相似（此示例在 A100 上运行）：

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



对于更大的模型，你可能更喜欢流式版本，它会在每个词元生成后立即将其打印出来：

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



&nbsp;

#### 专业提示 1：通过编译加快推理


如需获得最高 4× 的加速，可将

```python
model.to(device)
```

替换为

```python
model.to(device)
model = torch.compile(model)
```

注意：编译前期会产生数分钟的显著开销，并且要在第一次调用 `generate` 后才会体现加速效果。

下表展示了在 A100 上连续调用 `generate` 时的性能对比：

|                          | 硬件            | 令牌/秒    | 内存     |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B compiled | Nvidia A100 GPU | 107        | 1.99 GB  |


&nbsp;
<a id="pro-tip-2-speed-up-inference-with-compilation"></a>
#### 专业提示 2：使用 KV 缓存 加快推理

在 CPU 上运行模型时，使用带 KV 缓存 的 `Qwen3Model` 直接替代品可以显著提升推理性能。（有关 KV 缓存 的更多信息，请参阅我的文章[从零理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

请注意，由于更容易计算，表中只列出了 Nvidia CUDA 设备的峰值内存用量。不过，其他设备使用相似的精度格式，因此其内存用量可能也相近；在此生成 150 词元文本时，KV 缓存 的存储甚至会进一步降低内存用量。（不过，不同设备可能以不同方式实现矩阵乘法，因而峰值内存需求可能不同；上下文长度增加时，KV 缓存 的内存占用也可能大幅增长。）

| 模型            | 模式              | 硬件            | 令牌/秒    | GPU 内存（VRAM）   |
| --------------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model 0.6B | 普通              | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | 普通（已编译）    | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | KV 缓存          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model 0.6B | KV 缓存（已编译）| Mac Mini M4 CPU | 137        | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | 普通              | Mac Mini M4 GPU | 21         | -                 |
| Qwen3Model 0.6B | 普通（已编译）    | Mac Mini M4 GPU | 错误       | -                 |
| Qwen3Model 0.6B | KV 缓存          | Mac Mini M4 GPU | 28         | -                 |
| Qwen3Model 0.6B | KV 缓存（已编译）| Mac Mini M4 GPU | 错误       | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | 普通              | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | 普通（已编译）    | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV 缓存          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV 缓存（已编译）| Nvidia A100 GPU | 90         | 1.48 GB           |

请注意，以上所有设置都经过测试，生成的文本输出相同。



&nbsp;

#### 专业提示 3：批量推理

通过批量推理，我们可以进一步提高吞吐量。由于现在使用更多输入序列进行推理，这并非完全公平的对比；这种做法以增加内存使用为代价，提高了每秒处理的词元数。

只需对提示词的准备代码稍作修改即可。以下是一个批量提示词示例：

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存 版本的代码与之类似，但需要使用以下可直接替换的模块：

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


以下实验使用的批量大小为 8。

| 模型             | 模式              | 硬件            | 批量大小   | 令牌/秒   | GPU 内存（VRAM）   |
| ---------------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model  0.6B | 普通              | Mac Mini M4 CPU | 8          | 2          | -                 |
| Qwen3Model 0.6B  | 普通（已编译）    | Mac Mini M4 CPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV 缓存          | Mac Mini M4 CPU | 8          | 92         | -                 |
| Qwen3Model 0.6B  | KV 缓存（已编译）| Mac Mini M4 CPU | 8          | 128        | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | 普通              | Mac Mini M4 GPU | 8          | 36         | -                 |
| Qwen3Model 0.6B  | 普通（已编译）    | Mac Mini M4 GPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV 缓存          | Mac Mini M4 GPU | 8          | 61         | -                 |
| Qwen3Model 0.6B  | KV 缓存（已编译）| Mac Mini M4 GPU | 8          | -          | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | 普通              | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | 普通（已编译）    | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV 缓存          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV 缓存（已编译）| Nvidia A100 GPU | 8          | 280        | 1.75 GB           |

