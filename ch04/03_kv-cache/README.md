# 扩展材料：KV 缓存



**此文件夹演示如何为 GPT 模型添加 KV 缓存。**

&nbsp;
## 概述

简而言之，KV 缓存会存储中间的键（K）和值（V）计算结果，以便在推理时复用，从而显著加快响应生成速度。缺点是它会增加代码复杂度和内存用量，而且不能在训练期间使用。不过，部署 LLM 时，推理速度的提升通常足以抵消代码复杂度和内存方面的代价。

&nbsp;
## 工作原理

假设 LLM 正在生成文本。具体来说，假设 LLM 收到以下提示词："Time flies"。

下图使用第 3 章图示的修改版展示了底层注意力分数计算的一部分，并突出显示了键向量和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

如我们在第 2 章和第 4 章所学，LLM 每次生成一个词（或 token）。假设 LLM 生成了单词 "fast"，那么下一轮的提示词就变成了 "Time flies fast"。下图对此进行了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

对比前两幅图可以看出，前两个 token 对应的键向量和值向量完全相同，因此在每一轮生成下一个 token 时重新计算它们会造成浪费。

因此，KV 缓存的思路是实现一种缓存机制，存储此前生成的键向量和值向量以供复用，从而避免不必要的重复计算。

&nbsp;

## KV 缓存实现

KV 缓存有多种实现方式，其核心思路是在每个生成步骤中只为新生成的 token 计算键张量和值张量。

我选择了一种强调代码可读性的简单实现。直接浏览代码改动，是了解其实现方式最容易的方法。

此文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：从第 3 章和第 4 章整理出的自包含代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但加入了实现 KV 缓存所需的改动。

你可以选择以下任一方式：

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件，查找标记新增改动的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 使用你选择的文件差异比较工具查看这两个代码文件，比较其中的改动：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

下面简要介绍一下具体实现。

&nbsp;

### 1. 注册缓存缓冲区

我们在 `MultiHeadAttention` 构造函数中添加两个缓冲区 `cache_k` 和 `cache_v`，用于存放各个步骤累积拼接的键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 使用 `use_cache` 标志进行前向传播

接着，我们扩展 `MultiHeadAttention` 类的 `forward` 方法，使其接受 `use_cache` 参数。将新的一批 token 投影为 `keys_new`、`values_new` 和 `queries` 后，我们可以初始化 KV 缓存，也可以向现有缓存追加内容：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清空缓存

生成文本时，在相互独立的序列之间（例如，两次文本生成调用之间）必须重置这两个缓冲区，因此我们还在 `MultiHeadAttention` 类中添加了重置缓存的方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传递 `use_cache`

完成对 `MultiHeadAttention` 类的修改后，我们接着修改 `GPTModel` 类。首先，在构造函数中添加一个用于跟踪 token 位置的变量：

```python
self.current_pos = 0
```

然后，我们将单行的块调用替换为显式循环，并将 `use_cache` 传递给每个 Transformer 块：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述改动还要求我们稍微修改 `TransformerBlock` 类，使其接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，为了方便起见，我们在 `GPTModel` 中添加模型级重置方法，一次清空所有块的缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成过程中使用缓存

完成对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 的修改后，就可以在简单的文本生成函数中使用 KV 缓存了：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

注意，在步骤 c) 中，我们只通过 `logits = model(next_idx, use_cache=True)` 将新 token 输入模型。没有缓存时，由于没有已存储的键和值可供复用，我们会将整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)` 输入模型。

&nbsp;

## 简单的性能对比

从概念上了解 KV 缓存之后，一个关键问题是：在一个小例子中，它的实际表现如何？要试用这一实现，可以将前面提到的两个文件作为 Python 脚本运行；脚本会让这个 124 M 参数的小型 LLM 根据 4-token 提示词 "Hello, I am" 生成 200 个新 token：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备 M4 芯片的 Mac Mini（CPU）上，结果如下：

|                        | Token/秒 |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

由此可见，即使使用只有 124 M 参数的模型和长度仅为 200-token 的短序列，我们也已经获得 ~5x 加速。（请注意，这一实现优先考虑代码可读性，并未针对 CUDA 或 MPS 的运行速度进行优化；要实现这类优化，需要预分配张量，而不是反复实例化并拼接张量。）

**注意：** 两种情况下模型生成的都是“乱码”，文本看起来类似这样：

> 输出文本：Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章会训练该模型，之后你就可以在训练好的模型上使用 KV 缓存来生成连贯的文本（不过，KV 缓存仅用于推理）。这里使用未训练的模型，是为了让代码尽可能简单。

更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 两种实现生成的文本完全相同。这说明 KV 缓存实现正确，因为索引很容易出错，进而导致结果不一致。


&nbsp;

## KV 缓存的优缺点

随着序列长度增加，KV 缓存的优点和缺点会更加明显：

- [优点] **计算效率提高**：不使用缓存时，步骤 *t* 的注意力必须将新查询与此前的 *t* 个键进行比较，因此累计工作量按二次方增长，O(n²)。使用缓存后，每个键和值只计算一次，之后重复使用，从而使每步的总复杂度降为线性，O(n)。

- [缺点] **内存用量线性增长**：每个新 token 都会追加到 KV 缓存中。对于长序列和更大的 LLM，累积的 KV 缓存会变得很大，可能占用大量甚至难以承受的（GPU）内存。我们可以截断 KV 缓存作为权宜之计，但这样会进一步增加复杂度（不过，部署 LLM 时这样做也可能很值得。）



&nbsp;
## 优化 KV 缓存实现

上面的 KV 缓存概念实现主要面向代码可读性和教学用途，有助于清楚地展示思路；但在实际场景中部署时（尤其是模型更大、序列更长时），还需要更谨慎地优化。

&nbsp;
### 扩展缓存时的常见问题

- **内存碎片和重复分配**：像前面那样不断使用 `torch.cat` 拼接张量，会因频繁分配和重新分配内存而形成性能瓶颈。

- **内存用量线性增长**：如果处理不当，序列非常长时 KV 缓存会变得难以使用。

&nbsp;
#### 提示 1：预分配内存

我们可以根据预期的最大序列长度，预先分配足够大的张量，以代替反复拼接张量。这样可以确保内存用量稳定，并减少开销。伪代码如下：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

推理时，只需将数据写入这些预分配张量的切片即可。

&nbsp;
#### 提示 2：通过滑动窗口截断缓存

为了避免 GPU 内存占用过大，可以采用滑动窗口，并动态截断缓存。通过滑动窗口，我们只在缓存中保留最近的 `window_size` 个 token：


```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实际优化

你可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在配备 M4 芯片的 Mac Mini（CPU）上生成 200 个 token，并将窗口大小设为与上下文长度相同（以确保结果一致），代码运行时间对比如下：

|                                  | Token/秒 |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

遗憾的是，在 CUDA 设备上，这个小模型的加速优势会消失，因为设备间传输和通信的开销超过了 KV 缓存对这个小模型带来的收益。


&nbsp;
## 更多资源

1. [Qwen3 从零实现的 KV 缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 从零实现的 KV 缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从零开始理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) —— 本 README 的详细说明
