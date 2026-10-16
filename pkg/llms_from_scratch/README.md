# PyPI 软件包 `llms-from-scratch`

这个可选 PyPI 软件包可让你方便地导入《从零开始构建大型语言模型》一书各章节中的代码。

&nbsp;
## 安装

&nbsp;
### 从 PyPI 安装

从官方 [Python 软件包索引](https://pypi.org/project/llms-from-scratch/)（PyPI）安装 `llms-from-scratch` 软件包：

```bash
pip install llms-from-scratch
```

> **注意：**如果你使用 [`uv`](https://github.com/astral-sh/uv)，请用 `uv pip` 替换 `pip`，或者使用 `uv add`：

```bash
uv add llms-from-scratch
```



&nbsp;
### 从 GitHub 以可编辑模式安装

如果你想修改代码，并在开发过程中立即使用这些改动：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -e .
```

> **注意：**使用 `uv` 时，请运行：

```bash
uv add --editable . --dev
```



&nbsp;
## 使用此软件包

安装后，你可以使用以下方式导入任意章节中的代码：

```python
from llms_from_scratch.ch02 import GPTDatasetV1, create_dataloader_v1

from llms_from_scratch.ch03 import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    PyTorchMultiHeadAttention # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
)

from llms_from_scratch.ch04 import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTModelFast # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
    generate_text_simple
)

from llms_from_scratch.ch05 import (
    generate,
    train_model_simple,
    evaluate_model,
    generate_and_print_sample,
    assign,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    plot_losses,
    download_and_load_gpt2
)

from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    calc_accuracy_loader,
    evaluate_model,
    train_classifier_simple,
    plot_values,
    classify_review
)

from llms_from_scratch.ch07 import (
    download_and_load_file,
    format_input,
    InstructionDataset,
    custom_collate_fn,
    check_if_running,
    query_model,
    generate_model_scores
)

	
from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

from llms_from_scratch.appendix_d import find_highest_gradient, train_model
```



&nbsp;

### GPT-2 KV 缓存变体（附加材料）

```python
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

有关 KV 缓存的更多信息，请参阅 [KV 缓存自述文件](../../ch04/03_kv-cache)。



&nbsp;

### Llama  3（附加材料）

```python
from llms_from_scratch.llama3 import (
		load_weights_into_llama,
  	Llama3Model,
    Llama3ModelFast,
    Llama3Tokenizer,
    ChatFormat,
    clean_text
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

关于 `llms_from_scratch.llama3` 的用法，请参阅[此附加章节](../../ch05/07_gpt_to_llama/README.md)。

有关 KV 缓存的更多信息，请参阅 [KV 缓存自述文件](../../ch04/03_kv-cache)。


&nbsp;
### Qwen3（附加材料）

```python
from llms_from_scratch.qwen3 import (
    load_weights_into_qwen,
    Qwen3Model,
    Qwen3Tokenizer,
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple,
    generate_text_simple_stream
)

# KV cache drop-in replacements with batched inference support
from llms_from_scratch.kv_cache_batched.generate import (
    generate_text_simple,
    generate_text_simple_stream
)
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

关于 `llms_from_scratch.qwen3` 的用法，请参阅[此附加章节](../../ch05/11_qwen3/README.md)。

有关 KV 缓存的更多信息，请参阅 [KV 缓存自述文件](../../ch04/03_kv-cache)。
