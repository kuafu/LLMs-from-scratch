# Tiktoken 分词编码与底层架构技术解析文档

> **项目关联**：[`LLMs-from-scratch/gpt2`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2)  
> **本地源码路径**：[`tiktoken/`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken)  
> **本地缓存路径**：[`tiktoken_cache/`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken_cache)  
> **核心引用文件**：[`gpt2small.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/gpt2small.py) / [`tiktoken/_educational.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken/_educational.py) / [`tiktoken/load.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken/load.py)

---

## 目录
1. [项目集成与词库规格](#1-项目集成与词库规格)
2. [分词编码全流程细节](#2-分词编码全流程细节)
3. [核心算法细节：BPE 最小 Rank 贪心合并](#3-核心算法细节bpe-最小-rank-贪心合并)
4. [工程架构与源码结构](#4-工程架构与源码结构)
5. [词表来源、分发与本地缓存机制](#5-词表来源分发与本地缓存机制)
6. [性能内核：为什么 Tiktoken 极快](#6-性能内核为什么-tiktoken-极快)
7. [本地调试与学习指引](#7-本地调试与学习指引)

---

## 1. 项目集成与词库规格

当前工程实现了一个从零构建的 GPT-2 (124M) 模型。模型输入输出管线中的 Tokenizer 选型与词库配置如下：

### 1.1 词库基础信息
* **配置方式**：`tokenizer = tiktoken.get_encoding("gpt2")`
* **内部编码名称**：`r50k_base`
* **词表容量 (`vocab_size`)**：**`50257`**

### 1.2 词表容量构成公式
$$256 \text{ (基础单字节)} + 50,000 \text{ (BPE 学习合并 Token)} + 1 \text{ (特殊标记)} = 50,257$$

| 序号 | 组成部分 | Token ID 范围 | 说明 |
| :--- | :--- | :--- | :--- |
| **1** | **基础字节** | `0 ~ 255` | 覆盖单字节所有 256 种取值（`0x00 ~ 0xFF`），确保任意 Unicode/UTF-8 文本均可表达，**永远不会产生 `<unk>` 标记**。 |
| **2** | **BPE 合并项** | `256 ~ 50255` | 预训练阶段按出现频率由高到低迭代合并的 50,000 个高频字符对（如常用词、词根、前后缀）。 |
| **3** | **特殊标记** | `50256` | `<|endoftext|>`，用于标识文本序列的起始或截断。 |

### 1.3 模型维度对应关系
在 [`gpt2small.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/gpt2small.py#L192) 中，词表大小与神经网络两端严格对齐：
* **输入词嵌入**：`self.tok_emb = nn.Embedding(50257, 768)`
* **输出投影头**：`self.out_head = nn.Linear(768, 50257, bias=False)`

---

## 2. 分词编码全流程细节

`tiktoken` 处理文本分为四个清晰阶段：

```mermaid
flowchart TD
    A["输入原始文本 Raw String"] --> B{"1. 特殊标记检测与隔离<br>(Special Tokens Check)"}
    B -->|匹配特殊标记| S["直接输出固定特殊 Token ID<br>(如 <|endoftext|> -> 50256)"]
    B -->|普通文本片段| C["2. 正则预分词<br>(Regex Pre-tokenization)"]
    C --> D["拆分为独立分块 Chunks<br>(词、标点、缩写、数字)"]
    D --> E["3. 原始字节化<br>(Raw &[u8] 切片)"]
    E --> F["4. BPE 最小 Rank 贪心合并<br>(逐 Chunk 独立合并)"]
    F --> G["最终输出 Token IDs 序列"]
    S --> G
```

### 步骤 1：特殊标记隔离与安全性拦截
* **参数机制**：接受 `allowed_special` 与 `disallowed_special` 两个集合。
* **安全性机制**：当普通用户输入中出现未显式声明允许的特殊标记（如 prompt 中包含 `<|endoftext|>`），会直接抛出安全异常，从分词层面防止 Prompt 注入攻击。

### 步骤 2：正则预分词（Pre-tokenization）
为了避免不同词性、标点或数字被混乱合并，`tiktoken` 强制先利用正则对文本切块，**且规定 BPE 合并绝对不能跨越正则切出的 Chunk 边界**。

* **GPT-2 官方切分正则**：
  ```regex
  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  ```
* **正则规则解析**：
  * `'s|'t|'re|...`：英文缩写词单独成块，优先切出。
  * ` ?\p{L}+`：匹配带有前导可选空格的连续字母单词（如 `" hello"`）。
  * ` ?\p{N}+`：匹配带有前导可选空格的连续数字。
  * ` ?[^\s\p{L}\p{N}]+`：匹配标点及非字母非数字符号（标点与单词被严格隔开）。
  * `\s+(?!\S)|\s+`：独立匹配空白字符。

### 步骤 3：Raw Bytes 字节化
传统的 Python GPT-2 实现中维护了一个复杂的 `bytes_to_unicode` 打印字符映射表。而 `tiktoken` **直接抛弃了 Unicode 中间映射**，直接将正则切出的每个 Chunk 转换为标准的 UTF-8 字节切片 `&[u8]`，并在字节级别进行查找。

---

## 3. 核心算法细节：BPE 最小 Rank 贪心合并

BPE 词表（`mergeable_ranks`）本质上是一个哈希表：`HashMap<Vec<u8>, usize>`，键为字节序列，值为合并排名（Rank）。
* **Rank 越小** $\implies$ 在训练集中出现频次越高 $\implies$ 拥有更高的合并优先级。

### 3.1 编码执行算法逻辑

对每个 Chunk 内部的原始字节序列执行以下迭代：
1. **初始化**：将 Chunk 的每一个字节拆解为单字节切片 `parts = [b0, b1, b2, ...]`.
2. **扫描相邻切片对**：计算当前序列中所有相邻切片在 `mergeable_ranks` 字典中的值：
   $$\text{min\_rank} = \min_{0 \le i < \text{len}-1} \text{ranks.get}(parts[i] + parts[i+1])$$
3. **判定与合并**：
   * 若所有相邻对均不在词表中（即 `min_rank` 为空），则该 Chunk 编码结束。
   * 否则，选出 **Rank 最小（优先级最高）** 的相邻对 $(parts[k], parts[k+1])$，就地合并为单一新切片。
4. **循环直到无法合并**。
5. **查表映射**：最终剩下的各个切片直接查 `ranks` 转换为对应的整型 Token ID。

### 3.2 纯 Python 对照实现源码分析

在当前项目内置的教学源码 [`tiktoken/_educational.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken/_educational.py#L83-L116) 中，完整还原了这一算法：

```python
def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = "colour"
) -> list[int]:
    # 1. 拆分为基础字节列表
    parts = [bytes([b]) for b in input]
    
    while True:
        min_idx = None
        min_rank = None
        # 2. 遍历相邻对，寻找当前切片中 Rank 最小的相邻对
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # 3. 如果找不到任何可合并的对，说明已达到最简 token 粒度，合并完成
        if min_rank is None:
            break
        assert min_idx is not None

        # 4. 贪心合并该对，保持其余部分不变
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    # 5. 返回每个切片对应的 Token ID (即其最终 Rank)
    tokens = [mergeable_ranks[part] for part in parts]
    return tokens
```

---

## 4. 工程架构与源码结构

为了便于深度学习与调试，当前工程已将 `tiktoken` 源码完整平移至工程根目录下，项目文件树如下：

```text
LLMs-from-scratch/gpt2/
├── tiktoken/                           <-- 本地优先加载的源码包
│   ├── __init__.py                     <-- 包入口导出
│   ├── core.py                         <-- Python 封装层（Encoding 类）
│   ├── registry.py                     <-- 编码注册表（管理 "gpt2"、"cl100k_base"）
│   ├── load.py                         <-- 词表文件下载与缓存机制
│   ├── model.py                        <-- 模型名到编码名的映射
│   ├── _educational.py                 <-- 官方纯 Python 算法对照实现
│   └── _tiktoken.cp313-win_amd64.pyd   <-- Rust 编译的底层加速库（PyO3 绑定）
├── tiktoken_cache/                    <-- 本地词表缓存目录（离线完全自给）
├── gpt2small.py                        <-- GPT-2 核心训练与生成脚本
└── gtp2small.ipynb                     <-- 交互式 Notebook
```

### 4.1 导入优先级机制
在 Python 中，当前工作目录与脚本所在目录默认位于 `sys.path[0]`。
因此在当前工程执行 `import tiktoken` 时，解释器会**自动优先加载本地 `./tiktoken/`**，覆盖系统的全局 `site-packages`。

验证代码：
```python
import tiktoken
print(tiktoken.__file__)
# 输出: H:\LLM\BuildLLM\LLMs-from-scratch\gpt2\tiktoken\__init__.py
```

### 4.2 动静态库与源码对应关系
* **`.pyd` 文件本质**：`_tiktoken.cp313-win_amd64.pyd` 是 Windows 上的动态链接库（DLL），由 Rust 代码通过 `PyO3` 框架编译而成，内部封装了 C-Python ABI。
* **原始 Rust 源码仓库**：[https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
  * 对应核心文件：`src/lib.rs`（包含核心 `CoreBPE` 结构体和底层 `_byte_pair_merge` 函数）。

---

## 5. 词表来源、分发与本地缓存机制

### 5.1 训练语料来源
* **GPT-2 词表 (`gpt2` / `r50k_base`)**：
  * **原始语料**：OpenAI 的 **WebText** 数据集（约 40GB 文本，由 Reddit 上高赞外链网页爬取清洗而来）。
  * **训练演化**：以 256 个 UTF-8 基础字节为底，连续执行 50,000 次最高频相邻对合并，得到通用英文子词库。
* **GPT-4 词表 (`cl100k_base`)**：
  * 语料扩充至万亿级高质量多语言（强化中文）、开源代码、数学与科技论文，词表扩容至约 10 万。

### 5.2 官方网络托管与文件格式
`tiktoken` 首次使用时需通过网络获取词表，数据托管在微软 Azure Blob 存储：

| 编码名称 | 模型映射 | 远程下载路径 (Azure Blob) | SHA-256 校验和 |
| :--- | :--- | :--- | :--- |
| **`gpt2`** | 原版 GPT-2 | `vocab.bpe` & `encoder.json`<br>🔗 [openaipublic.../gpt-2/encodings/main/vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe) | `1ce1664...` / `1961396...` |
| **`r50k_base`** | GPT-2 / GPT-3 | 🔗 [openaipublic.../encodings/r50k_base.tiktoken](https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken) | `306cd27...` |
| **`cl100k_base`** | GPT-4 / 3.5 | 🔗 [openaipublic.../encodings/cl100k_base.tiktoken](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) | `223921b...` |

* **`.tiktoken` 存储格式**：极其精简，每一行均为 `Base64(Token 原始字节) + " " + Rank`。

### 5.3 缓存目录改造实现（由系统临时转本地工程）

原生 `tiktoken` 默认将缓存文件存放于系统的临时目录（`%TEMP%\data-gym-cache`）。为保证工程的**独立性、可移植性与离线可用性**，本项目修改了本地 [`tiktoken/load.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken/load.py#L41-L46)：

```python
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        # 默认回落到当前工程根目录下的 tiktoken_cache 文件夹
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "tiktoken_cache")
        user_specified_cache = True
```

### 5.4 缓存文件哈希身份解密（关键认知）

在 [`tiktoken_cache/`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken_cache) 目录下，文件名是**下载 URL 的 SHA-1 哈希值**。其各文件的对应关系如下：

| 缓存文件名 (URL SHA-1) | 文件大小 | 对应真实文件 | 作用与模型归属 |
| :--- | :--- | :--- | :--- |
| **`6d1cbeee0f20b3d9...`** | 456 KB | `vocab.bpe` | **GPT-2 核心**：BPE 规则合并先后顺序列表 |
| **`6c7ea1a7e38e3a7f...`** | 1.0 MB | `encoder.json` | **GPT-2 核心**：字节字符串到 Token ID 的全量映射表 |
| **`9b5ad71b2ce53022...`** | 1.68 MB | `cl100k_base.tiktoken` | **GPT-4 / ChatGPT 核心**：10 万词表文件 |
| **`0ea1e91bbb3a60f7...`** | 835 KB | `r50k_base.tiktoken` | **GPT-2/3 标准版**：单文件版的 50k 词表 |

> [!NOTE]
> **为什么当前工程默认只需要前 2 个文件？**  
> 因为当前项目 [`gpt2small.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/gpt2small.py#L44) 明确使用 `tiktoken.get_encoding("gpt2")`。该实现完全基于原版 GPT-2 的 `vocab.bpe` 与 `encoder.json` 两个文件。只要拥有这两个文件，当前 GPT-2 模型的训练与推理就是 **100% 完整且可完全脱网离线运行** 的。第 3 个文件为 GPT-4 词表。

---

## 6. 性能内核：为什么 Tiktoken 极快

相较于早期原版 Python 分词器或部分纯解释型库，`tiktoken` 拥有 5~30 倍的速度优势，核心来自以下底层工程优化：

1. **Rust 原生内存零拷贝**：
   无需在 Python 解释器和堆内存之间频繁创建字符串对象，直接基于原生字节数组切片 `&[u8]` 计算哈希与对比。
2. **多线程并发切分（Rayon 驱动）**：
   由于正则预分词切分出来的各个 Chunk 之间完全互不影响，Rust 内核利用 `rayon` 库，把整篇长文本中的不同 Chunk 自动并发分发到 CPU 多个物理核心上并行做 BPE 合并。
3. **基于索引跳跃的局部增量合并**：
   常规实现每次合并后重新扫描整串切片，复杂度为 $O(N^2)$。`tiktoken` 内部维护了类似双向链表的紧凑索引数组，合并某一位置 $(i, i+1)$ 后，仅需更新其前驱 $(i-1, i)$ 与后继 $(i, i+2)$ 的合并得分，避免全量重新扫描。

---

## 7. 本地调试与学习指引

拥有本地源码与本地缓存后，您可以非常方便地进行单步调试和内部机制观测：

### 技巧 1：开启算法可视化合并过程
在工程中直接调用教学模块，观测 `"Hello world"` 是如何一步步被字节合并的：

```python
from tiktoken._educational import SimpleBytePairEncoding
import tiktoken

# 从当前工程加载 gpt2 规则
enc = SimpleBytePairEncoding.from_tiktoken("gpt2")

# 开启可视化（将在控制台打印合并过程）
tokens = enc.encode("Hello world", visualise="simple")
print("Tokens:", tokens)
```

### 技巧 2：在 `core.py` 中追踪内部参数
在 [`tiktoken/core.py`](file:///h:/LLM/BuildLLM/LLMs-from-scratch/gpt2/tiktoken/core.py) 中的 `encode` 方法加入调试输出：
```python
def encode(self, text: str, *, allowed_special="all", disallowed_special=()):
    print(f"[Debug] Encoding text of length: {len(text)}")
    # ... 原有底层调用 ...
```
即可在不影响全局 Python 环境的前提下，深度观测工程训练和推理时的实际分词行为。
