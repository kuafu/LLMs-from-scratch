# 从零实现 Tiny Aya 3.35B

Tiny Aya 是 Cohere 推出的一款新型“小型” LLM，据称是 3B 参数规模中“能力最强的多语言开放权重模型”。（根据[公告文章](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 的表现优于 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



这是一个很适合在本地运行和试验的模型。唯一需要注意的是，尽管它是开放权重模型，但其许可条款限制较多，只允许非商业用途。

除此之外，Arya 是一个 3.35B 参数模型，提供了多个版本，适合个人和（非商业）研究用途：

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base)（基础模型）
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)（在不同语言和地区间取得最佳平衡；笔记本默认版本）
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)（针对南亚语言优化）
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)（针对欧洲和亚太语言优化）
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)（针对西亚和非洲语言优化）



具体而言，以下列出了这些模型针对哪些语言进行了优化：

| 地区             | 语言                                                         | 优化模型        |
| ---------------- | ------------------------------------------------------------ | --------------- |
| **亚太**         | 繁体中文、粤语、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印度尼西亚语、简体中文、日语 | tiny-aya-water  |
| **非洲**         | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、修纳语、约鲁巴语、尼日利亚皮钦语、马达加斯加语 | tiny-aya-earth  |
| **南亚**         | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire   |
| **欧洲**         | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、博克马尔语、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water  |
| **西亚**         | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语             | tiny-aya-earth  |


从架构上看，Tiny Aya 是一种经典的解码器式 Transformer，并做了一些值得注意的修改（除了 SwiGLU 和分组查询注意力等显而易见的改动之外）：

1. **并行 Transformer 块。**并行 Transformer 块基于同一个经过归一化的输入计算注意力和 MLP，然后一步将两者都加到残差上。我认为这样做是为了减少层内的串行依赖，从而提高计算吞吐量。

2. **滑动窗口注意力。**具体来说，它采用 3:1 的局部与全局比例，类似于 Arcee Trinity 和 Olmo 3。窗口大小也是 4096。与 Arcee 类似，滑动窗口层使用 RoPE，而完整注意力层使用 NoPE。

3. **LayerNorm。**大多数架构都改用了 RMSNorm，因为它的计算开销略低且效果良好。Tiny Aya 则保留了更经典的做法，使用经过修改的 LayerNorm（此处的实现类似标准 LayerNorm，但没有平移项，也就是偏置参数）。



&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) 是一个独立的 Jupyter 笔记本，实现了 Tiny Aya 架构并加载预训练权重。


另一个笔记本 [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) 加入了 KV 缓存，以提升运行时性能（但会增加代码复杂度）。有关 KV 缓存 的更多信息，请参阅我的文章[从零理解并编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。


<br>

有关架构差异以及与其他架构的对比，请参阅我的文章[《大型 LLM 架构对比：从 DeepSeek-V3 到 Kimi K2，纵览现代 LLM 架构设计》](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。





