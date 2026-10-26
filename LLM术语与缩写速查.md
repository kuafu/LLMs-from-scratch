# LLM 术语与缩写速查

这份表按本仓库的学习路线整理重点名词：文本与分词、注意力、GPT 架构、预训练、生成、微调、评估和运行性能。每条都给出常见英文写法、简要解释及可继续阅读的仓库资料。英文栏中的大写字母只有在业内确实作为缩写使用时才按缩写理解；`Q`、`K`、`V` 等是代码和公式中的记号。

本仓库的中文内容有时将 *token* 译为“词元”“标记”或“令牌”，下文统一用“词元”。“参数”“权重”“梯度”等词在不同上下文有更严格的数学含义，这里先给出理解代码所需的含义。

## 1. 基础概念与文本数据

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 大语言模型 | Large Language Model（LLM） | 在大量文本上训练、以词元序列为输入并生成或处理文本的语言模型。 | [总览][repo] |
| 自然语言处理 | Natural Language Processing（NLP） | 让计算机分析、理解和生成自然语言的研究与应用领域。 | [总览][repo] |
| 语言模型 | Language Model（LM） | 为词元序列或下一个词元分配概率的模型；LLM 是其中规模较大的一类。 | [第 4 章][ch04] |
| 生成式预训练 Transformer | Generative Pre-trained Transformer（GPT） | 本仓库实现的模型家族：基于 Transformer 解码器，以自回归方式预测后续词元。 | [第 4 章][ch04] |
| 语料 | corpus | 用于训练或评估的一组文本；语料质量和分布会影响模型行为。 | [第 2 章][ch02] |
| 词元 | token | 分词器处理文本得到的离散单位，可以是字、子词、标点或特殊符号。 | [第 2 章][ch02] |
| 分词 / 词元化 | tokenization | 按分词器规则把文本切分成词元，并将其映射为可供模型处理的 ID。 | [第 2 章][ch02] |
| 分词器 | tokenizer | 执行文本与词元 ID 之间编码、解码的规则和实现。 | [第 2 章][ch02] |
| 词元 ID | token ID | 词元在词表中的整数编号；模型接收的是 ID，而非字符串本身。 | [第 2 章][ch02] |
| 词表 / 词汇表 | vocabulary（vocab） | 分词器可表示的词元及其 ID 的集合。 | [第 2 章][ch02] |
| 词表大小 | vocabulary size | 词表中的词元数；决定输入嵌入和输出预测的相关维度。 | [第 2 章][ch02] |
| 未知词元 | unknown token（`[UNK]`） | 当分词器不能表示输入片段时使用的占位词元；并非所有分词器都需要它。 | [第 2 章][ch02] |
| 特殊词元 | special token | 用于标记边界、角色、填充等控制信息的词元，不按普通文本片段解释。 | [第 2 章][ch02] |
| 序列开始标记 | beginning-of-sequence token（BOS） | 标识一个序列的起始；具体分词器不一定单独使用此标记。 | [第 2 章][ch02] |
| 序列结束标记 | end-of-sequence token（EOS） | 标识序列结束；GPT-2 的 endoftext 特殊词元可承担类似用途。 | [第 2 章][ch02] |
| 填充标记 | padding token（PAD） | 为短序列补齐批次长度的占位词元；部分实现复用结束标记。 | [第 2 章][ch02] |
| 字节对编码 | Byte Pair Encoding（BPE） | 反复合并高频相邻单元以构造子词词表的分词方法。 | [BPE 专题][bpe] |
| 子词 | subword | 介于字符和完整单词之间、可复用的分词单位；有助于处理罕见词。 | [BPE 专题][bpe] |
| 编码 / 解码 | encode / decode | 分词器的两个方向：文本转词元 ID，以及词元 ID 转回文本。 | [第 2 章][ch02] |
| 上下文长度 / 上下文窗口 | context length / context window | 模型一次处理的词元序列长度；超过限制时需要截断或分段。 | [第 2 章][ch02] |
| 序列长度 | sequence length | 某个输入批次或样本实际包含的词元位置数。 | [第 2 章][ch02] |
| 滑动窗口采样 | sliding-window sampling | 从长文本按窗口和步幅抽取相邻训练片段，用前面的词元预测后面的词元。 | [第 2 章][ch02] |
| 步幅 | stride | 滑动窗口每次向前移动的词元数；影响相邻训练样本的重叠程度。 | [第 2 章][ch02] |
| 输入与目标序列 | input and target sequences | 下一个词元预测中，目标序列通常是输入序列整体右移一位。 | [第 2 章][ch02] |
| 词元嵌入 | token embedding | 把离散词元 ID 映射到可训练的连续向量。 | [第 2 章][ch02] |
| 位置嵌入 | positional embedding | 为序列位置提供信息，使模型能区分相同词元在不同位置的出现。 | [第 2 章][ch02] |
| 嵌入维度 | embedding dimension | 每个嵌入向量的分量数，在代码中常写作 `emb_dim`。 | [第 2 章][ch02] |
| 批次 | batch | 一次并行送入模型的一组样本；批次大小是其中的样本数。 | [第 2 章][ch02] |
| 数据加载器 | DataLoader | 按批次读取、打乱并迭代训练或评估样本的工具。 | [第 2 章][ch02] |

## 2. 注意力机制与上下文建模

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 注意力机制 | attention mechanism | 根据当前位置与其他位置的相关性，对信息加权汇总。 | [第 3 章][ch03] |
| 自注意力 | self-attention | 查询、键和值都来自同一输入序列的注意力机制。 | [第 3 章][ch03] |
| 因果注意力 | causal attention | 遮蔽未来位置，使当前位置只利用自身及之前的词元。 | [第 3 章][ch03] |
| 查询 | query（Q） | 当前待更新位置用于匹配其他位置的向量。 | [第 3 章][ch03] |
| 键 | key（K） | 每个可被关注位置用于与查询计算匹配程度的向量。 | [第 3 章][ch03] |
| 值 | value（V） | 按注意力权重加权汇总、传给当前查询位置的向量。 | [第 3 章][ch03] |
| 查询 / 键 / 值投影 | Q / K / V projections | 通过可训练线性层把输入表示映射为查询、键和值。 | [第 3 章][ch03] |
| 注意力分数 | attention score | 查询与键的匹配值；经缩放、掩码和归一化后得到注意力权重。 | [第 3 章][ch03] |
| 注意力权重 | attention weight | 归一化后的分数，决定各值向量对当前位置输出的贡献。 | [第 3 章][ch03] |
| 缩放点积注意力 | Scaled Dot-Product Attention（SDPA） | 用 Q 与 K 的点积衡量匹配，并通常除以键维度的平方根来稳定数值。 | [高效注意力专题][efficient-attn] |
| Softmax | softmax | 将一组分数转成非负且总和为 1 的权重或概率。 | [第 3 章][ch03] |
| 注意力掩码 | attention mask | 在计算权重前排除不应关注的位置，例如未来词元。 | [第 3 章][ch03] |
| 因果掩码 | causal mask | 注意力掩码的一种，阻止当前位置看到未来位置。 | [第 3 章][ch03] |
| 注意力头 | attention head | 一组独立的 Q/K/V 投影与注意力计算；不同头可学习不同关系。 | [第 3 章][ch03] |
| 多头注意力 | Multi-Head Attention（MHA） | 并行使用多个注意力头，再将结果拼接并映射回模型维度。 | [第 3 章][ch03] |
| 头维度 | head dimension | 单个注意力头中 Q/K/V 向量的维度，通常与总维度和头数相关。 | [第 3 章][ch03] |
| FlashAttention | FlashAttention | 一类注重减少显存读写的精确注意力计算实现；仓库扩展将其与其他 MHA 实现比较。 | [高效注意力专题][efficient-attn] |
| FlexAttention | FlexAttention | 支持用可编程规则定义注意力掩码或分数修饰的实现接口。 | [高效注意力专题][efficient-attn] |
| 分组查询注意力 | Grouped-Query Attention（GQA） | 多个查询头共享同一组键头和值头，以降低 KV 缓存规模。 | [GQA 专题][gqa] |
| 多查询注意力 | Multi-Query Attention（MQA） | 所有查询头共享一组键头和值头，可视为 GQA 的共享极端。 | [GQA 专题][gqa] |
| 多头潜在注意力 | Multi-Head Latent Attention（MLA） | 使用低维潜在表示压缩键值相关信息，减少推理时的缓存需求。 | [MLA 专题][mla] |
| 滑动窗口注意力 | Sliding Window Attention（SWA） | 每个位置只关注有限的局部历史窗口，控制长序列计算与缓存开销。 | [SWA 专题][swa] |
| DeepSeek 稀疏注意力 | DeepSeek Sparse Attention（DSA） | 用可学习的选择机制选取部分历史词元；仓库示例仍构造稠密分数矩阵，主要演示选择逻辑。 | [DSA 专题][dsa] |
| 门控 DeltaNet | Gated DeltaNet | 仓库中的线性注意力变体，利用门控与状态更新处理历史信息。 | [DeltaNet 专题][deltanet] |
| Kimi Delta 注意力 | Kimi Delta Attention（KDA） | 门控 DeltaNet 的逐通道门控变体，见仓库中的 Kimi Linear 扩展。 | [DeltaNet 专题][deltanet] |
| KV 缓存 | key-value cache（KV cache） | 推理时保存已有词元计算出的键和值，生成下一个词元时复用。 | [KV 缓存专题][kv] |
| 跨层 KV 共享 | cross-layer KV sharing | 让不同 Transformer 层复用部分 KV 信息，以进一步减少缓存。 | [跨层共享专题][crosskv] |

## 3. 模型结构与张量

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| Transformer | Transformer | 以注意力和前馈网络为主要组成部分的模型架构。 | [第 4 章][ch04] |
| 解码器式模型 | decoder-only model | 仅堆叠 Transformer 解码器块的语言模型结构；本仓库的 GPT 属于此类。 | [第 4 章][ch04] |
| 自回归模型 | autoregressive model | 按顺序生成词元，每一步以前面已知词元为条件预测下一个。 | [第 4 章][ch04] |
| 编码器式模型 | encoder-based model | 为整个输入构建表示的结构，常用于理解或分类任务；仓库 IMDb 扩展与解码器式模型作比较。 | [IMDb 扩展][imdb] |
| 参数 / 权重 | parameter / weight | 训练中更新的模型数值，线性层矩阵与嵌入矩阵是常见例子。 | [第 4 章][ch04] |
| 隐藏状态 / 隐藏表示 | hidden state / representation | 模型中间层为每个词元计算的向量表示。 | [第 4 章][ch04] |
| 张量 | tensor | PyTorch 中带形状和数据类型的多维数值数组。 | [附录 A][appendix-a] |
| 张量形状 | tensor shape | 张量各维度的长度，例如批次、序列长度、嵌入维度。 | [附录 A][appendix-a] |
| 计算图 | computational graph | 记录张量运算依赖关系的数据结构，供自动求梯度使用。 | [附录 A][appendix-a] |
| 自动微分 | automatic differentiation / autograd | 根据计算图自动计算损失对可训练参数的梯度。 | [附录 A][appendix-a] |
| 线性层 | linear layer | 使用权重矩阵（及可选偏置）对输入向量做线性变换。 | [第 4 章][ch04] |
| 前馈网络 | Feed-Forward Network（FFN） | 每个词元位置独立通过若干线性层与激活函数的子网络。 | [第 4 章][ch04] |
| 多层感知机 | Multi-Layer Perceptron（MLP） | 由多层线性变换和非线性激活构成；在本仓库常指 Transformer 块的前馈部分。 | [第 4 章][ch04] |
| 激活函数 | activation function | 为线性变换引入非线性的函数。 | [第 4 章][ch04] |
| GELU | Gaussian Error Linear Unit（GELU） | GPT 前馈层使用的平滑非线性激活函数。 | [第 4 章][ch04] |
| ReLU | Rectified Linear Unit（ReLU） | 将负输入置零、正输入保留的分段线性激活函数。 | [第 4 章][ch04] |
| SiLU / Swish | Sigmoid Linear Unit（SiLU）/ Swish | 利用输入与其 sigmoid 值相乘的平滑激活函数。 | [Llama 专题][llama] |
| 门控线性单元 | Gated Linear Unit（GLU） | 通过门控分支控制另一分支输出的结构。 | [Llama 专题][llama] |
| SwiGLU | Swish-Gated Linear Unit（SwiGLU） | 用 Swish 激活与门控分支组合的前馈层变体。 | [第 4 章][ch04] |
| 层归一化 | Layer Normalization（LayerNorm） | 对单个词元的特征维度做归一化，帮助稳定深层网络的计算。 | [第 4 章][ch04] |
| 均方根归一化 | Root Mean Square Normalization（RMSNorm） | 依据特征的均方根缩放表示；多种扩展模型采用。 | [Llama 专题][llama] |
| 捷径 / 残差连接 | shortcut / residual connection | 将子层输入与输出相加，为深层网络提供直接的信息与梯度路径。 | [第 4 章][ch04] |
| Transformer 块 | Transformer block | 组合注意力、前馈网络、归一化和残差连接的重复计算单元。 | [第 4 章][ch04] |
| 并行 Transformer 块 | parallel Transformer block | 基于同一归一化输入并行计算注意力与前馈分支，再将两者结果加入残差。 | [Tiny Aya 专题][tiny-aya] |
| 线性 / 完整注意力混合 | linear / full attention hybrid | 在模型不同层组合线性注意力与完整注意力，平衡表示能力和计算开销。 | [Qwen3.5 专题][qwen35] |
| 缓冲区 | PyTorch buffer | 附属于模块但不由优化器更新的张量，例如因果掩码；可随模块一起迁移设备。 | [缓冲区专题][buffers] |
| 输出头 / 语言模型头 | output head / LM head | 将隐藏表示映射成词表上各候选词元的预测分数。 | [第 4 章][ch04] |
| 权重绑定 | weight tying | 让输出预测层与输入词元嵌入层共享权重矩阵。 | [第 4 章][ch04] |
| 旋转位置嵌入 | Rotary Position Embedding（RoPE） | 通过旋转 Q、K 向量编码相对位置信息的方案。 | [Llama 专题][llama] |
| 混合专家 | Mixture of Experts（MoE） | 由路由器为每个词元选择部分专家前馈网络进行计算。 | [MoE 专题][moe] |
| 专家 | expert | MoE 中可被路由器选中的一个子网络，通常是前馈网络。 | [MoE 专题][moe] |
| 路由器 / 门控 | router / gate | 根据词元表示决定激活哪些专家及其组合权重。 | [MoE 专题][moe] |

## 4. 预训练与优化

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 预训练 | pretraining | 在大规模通用文本上先学习语言规律，再用于生成或后续任务。 | [第 5 章][ch05] |
| 自监督学习 | self-supervised learning | 从原始数据本身构造训练目标；语言模型的下一个词元预测就是一例。 | [第 5 章][ch05] |
| 下一个词元预测 | next-token prediction | 给定前文，为下一个位置的正确词元提高预测概率。 | [第 5 章][ch05] |
| 训练集 | training set | 用于计算梯度并更新模型参数的数据。 | [第 5 章][ch05] |
| 验证集 | validation set | 训练期间用于观察泛化与调参的数据，不参与该步参数更新。 | [第 5 章][ch05] |
| 测试集 | test set | 模型与调参确定后用于独立评估的数据。 | [第 6 章][ch06] |
| 前向传播 | forward pass | 根据当前参数和输入计算模型输出及损失。 | [第 5 章][ch05] |
| 反向传播 | backpropagation / backward pass | 从损失出发求各可训练参数的梯度。 | [第 5 章][ch05] |
| 梯度 | gradient | 损失对参数的导数，指示小幅参数变化对损失的影响。 | [第 5 章][ch05] |
| 优化器 | optimizer | 根据梯度和优化规则更新参数的算法或实现。 | [第 5 章][ch05] |
| AdamW | AdamW | 本仓库主要训练脚本使用的优化器，结合自适应更新与解耦权重衰减。 | [第 5 章][ch05] |
| Muon | Muon | 扩展材料中用于部分二维权重矩阵的优化器；其他参数仍可交给 AdamW。 | [Muon 专题][muon] |
| 学习率 | learning rate（LR） | 控制单次参数更新幅度的超参数。 | [第 5 章][ch05] |
| 学习率调度 | learning-rate schedule | 随训练进度改变学习率的规则。 | [附录 D][appendix-d] |
| 线性预热 | linear warmup | 在训练初期逐步提高学习率，以降低一开始更新过大的风险。 | [训练循环扩展][scheduler] |
| 余弦衰减 | cosine decay | 让学习率按余弦曲线逐步降低的调度方法。 | [训练循环扩展][scheduler] |
| 权重衰减 | weight decay | 优化时对参数大小施加约束的机制，通常用于改善泛化。 | [第 5 章][ch05] |
| 梯度裁剪 | gradient clipping | 当梯度范数过大时缩小梯度，限制单步更新的幅度。 | [附录 D][appendix-d] |
| L2 范数 | L2 norm / Euclidean norm | 向量大小的度量；梯度裁剪可限制所有梯度的整体 L2 范数。 | [附录 D][appendix-d] |
| 训练步 / 更新步 | training step | 完成一次参数更新的训练迭代；与处理一个批次常相对应。 | [第 5 章][ch05] |
| 轮次 | epoch | 对训练集完成一次遍历；若数据被分批，一轮包含多个训练步。 | [第 5 章][ch05] |
| 损失函数 | loss function | 衡量模型预测与训练目标差距的可优化数值。 | [第 5 章][ch05] |
| 交叉熵损失 | cross-entropy loss | 根据正确词元的预测概率计算损失；概率越低，惩罚越大。 | [第 5 章][ch05] |
| 困惑度 | perplexity（PPL） | 语言模型预测词元难度的指标，通常为平均交叉熵的指数。 | [第 5 章][ch05] |
| 过拟合 | overfitting | 在训练集上变好、但在未见数据上的效果未同步改善。 | [第 5 章][ch05] |
| Dropout | dropout | 训练时随机置零部分激活，以帮助降低过拟合风险。 | [第 3 章][ch03] |
| 检查点 | checkpoint | 保存模型参数及必要训练状态的文件，用于恢复或部署。 | [第 5 章][ch05] |
| 状态字典 | `state_dict` | PyTorch 中保存参数和持久缓冲区的键值映射，可用于保存及加载模型状态。 | [第 5 章][ch05] |
| 模型权重加载 | weight loading | 将已训练参数读入与之匹配的模型结构。 | [第 5 章][ch05] |
| 内存映射加载 | memory-mapped I/O（mmap） | 从磁盘按需映射权重数据，降低一次性载入整个文件的内存需求。 | [高效加载专题][mmap] |
| GaLore | Gradient Low-Rank Projection（GaLore） | 对梯度进行低秩投影的优化思路；仓库将其列为可选预训练改进。 | [Gutenberg 扩展][gutenberg] |
| 随机种子 | random seed | 初始化随机数生成器的数值，有助于复现实验；不同设备仍可能有差异。 | [CPU/MPS 差异][cpu-mps] |

## 5. 推理与文本生成

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 推理 | inference | 使用已训练的参数处理输入或生成结果，不执行训练更新。 | [第 5 章][ch05] |
| 提示词 | prompt | 提供给模型的输入文本或消息，作为后续生成的上下文。 | [第 5 章][ch05] |
| 输出续写 | completion | 模型接在提示词之后生成的文本。 | [第 5 章][ch05] |
| 预填充 | prefill | 生成开始前处理已有提示词的阶段；使用 KV 缓存时可保存其键和值。 | [KV 缓存专题][kv] |
| 逐词元解码 | decoding | 基于当前上下文反复预测、选取并追加下一个词元。 | [第 5 章][ch05] |
| Logits | logits | Softmax 之前、为词表中每个候选词元计算的原始分数。 | [第 5 章][ch05] |
| 预测概率 | predicted probability | 对 logits 进行 Softmax 后得到的候选词元概率分布。 | [第 5 章][ch05] |
| 贪心解码 | greedy decoding | 每一步选择概率最高的词元，输出确定但可能重复或欠缺多样性。 | [第 5 章][ch05] |
| 随机采样 | sampling | 按候选词元的概率分布随机选取下一个词元。 | [第 5 章][ch05] |
| 温度 | temperature | 在 Softmax 前缩放 logits；较低值使分布更集中，较高值使其更平坦。 | [第 5 章][ch05] |
| Top-k 采样 | top-k sampling | 每一步仅保留概率最高的 `k` 个候选词元，再从中采样。 | [第 5 章][ch05] |
| 最大新词元数 | maximum new tokens | 一次生成允许新增的词元上限。 | [第 5 章][ch05] |
| 响应延迟 | latency | 从发起请求到获得输出所需的时间，可能包括预填充和逐词元解码。 | [KV 缓存专题][kv] |
| 生成吞吐量 | generation throughput | 单位时间生成的词元数，常记作 tokens/s。 | [GQA 专题][gqa] |
| 训练吞吐量 | training throughput | 单位时间完成的训练工作量，仓库以每秒处理词元数比较方案。 | [训练加速专题][train-speed] |

## 6. 任务微调与指令数据

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 微调 | fine-tuning | 在预训练模型基础上继续训练，使其适应特定任务或行为。 | [第 6 章][ch06] |
| 全量微调 | full fine-tuning | 更新预训练模型的全部原有参数；相对 LoRA 等方法成本更高。 | [附录 E][lora] |
| 监督微调 | Supervised Fine-Tuning（SFT） | 使用有目标答案的样本，通过监督损失训练模型。 | [第 7 章][ch07] |
| 指令微调 | instruction fine-tuning | 使用指令与期望回答组成的数据，使模型更善于遵循用户要求。 | [第 7 章][ch07] |
| 任务专用微调 | task-specific fine-tuning | 针对分类等具体任务调整预训练模型。 | [第 6 章][ch06] |
| 分类 | classification | 从预定义类别中选择一个类别；第 6 章以垃圾邮件识别为例。 | [第 6 章][ch06] |
| 类别标签 | class label | 分类样本的正确类别，训练时与模型预测比较。 | [第 6 章][ch06] |
| 分类头 | classification head | 将模型隐藏表示映射到类别分数的输出层。 | [第 6 章][ch06] |
| 分类准确率 | classification accuracy | 预测正确的样本数占评估样本总数的比例。 | [第 6 章][ch06] |
| 冻结参数 | frozen parameters | 训练时保留原有数值、不计算或不应用其梯度更新的参数。 | [第 6 章][ch06] |
| 部分微调 | partial fine-tuning | 冻结大部分原有层，只更新输出头或选定的层。 | [第 6 章][ch06] |
| 类别不平衡 | class imbalance | 不同类别的样本数差异较大，可能使分类器偏向多数类。 | [第 6 章][ch06] |
| 欠采样 | undersampling | 减少多数类样本数以缓解类别不平衡。 | [第 6 章][ch06] |
| 末词元表示 | last-token representation | 因果模型最后位置的隐藏状态可汇总此前词元的信息，供分类头使用。 | [第 6 章][ch06] |
| 指令 / 输入 / 响应 | instruction / input / response | 指令数据集中的任务要求、可选补充内容和期望输出。 | [第 7 章][ch07] |
| 提示模板 | prompt template | 将指令、可选输入和回答标记组织成固定文本格式的规则。 | [第 7 章][ch07] |
| 填充 | padding | 为了组成批次，在较短序列末尾加入占位词元。 | [第 7 章][ch07] |
| 截断 | truncation | 当文本超过模型或批次允许长度时去掉超出部分的词元。 | [第 6 章][ch06] |
| 动态填充 | dynamic padding | 按当前批次最长序列补齐，减少无效填充位置。 | [第 7 章][ch07] |
| 损失掩码 | loss mask | 让某些位置不参与损失计算；可用来只对回答部分训练。 | [第 7 章][ch07] |
| 只计算回答损失 | response-only loss | 忽略指令与填充位置的目标，仅用回答词元更新模型；仓库练习展示这一变体。 | [第 7 章练习][ch07ex] |
| 参数高效微调 | Parameter-Efficient Fine-Tuning（PEFT） | 只训练少量新增或选定参数以适应任务的方法类别。 | [附录 E][lora] |
| 低秩自适应 | Low-Rank Adaptation（LoRA） | 冻结原权重，用两个较小矩阵近似权重更新量。 | [附录 E][lora] |
| LoRA 秩 | LoRA rank（`r`） | 低秩矩阵中间维度，影响可训练参数量和更新表达能力。 | [附录 E][lora] |
| LoRA 缩放系数 | LoRA scaling（`alpha`） | 控制低秩更新相对原权重贡献的参数。 | [附录 E][lora] |
| 知识蒸馏 | knowledge distillation | 用较大模型的输出指导较小模型训练；仓库 IMDb 扩展中提到蒸馏模型。 | [IMDb 扩展][imdb] |

## 7. 偏好对齐与评估

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 对齐 | alignment | 调整模型行为，使其更符合给定指令、偏好或任务要求。 | [DPO 专题][dpo] |
| 偏好数据 | preference data | 对同一个提示词给出偏好较高和较低的回答，用于比较学习。 | [DPO 专题][dpo] |
| 首选回答 | chosen response（`chosen`） | 偏好数据中被认为更好的回答。 | [DPO 专题][dpo] |
| 非首选回答 | rejected response（`rejected`） | 与首选回答对比、偏好较低的回答。 | [DPO 专题][dpo] |
| 偏好微调 | preference fine-tuning | 利用偏好比较信号继续调整已具备指令能力的模型。 | [DPO 专题][dpo] |
| 直接偏好优化 | Direct Preference Optimization（DPO） | 利用首选/非首选回答及参考模型概率，直接优化语言模型的偏好目标。 | [DPO 专题][dpo] |
| 策略模型 | policy model（πθ） | 偏好优化中正在更新的语言模型。 | [DPO 专题][dpo] |
| 参考模型 | reference model（πref） | 作为比较基准、在 DPO 训练中保持固定的模型。 | [DPO 专题][dpo] |
| DPO 的 β | beta（β） | 调节策略模型相对参考模型偏离程度的超参数。 | [DPO 专题][dpo] |
| 奖励模型 | reward model | 为回答质量打分的模型；DPO 训练不要求单独训练显式奖励模型。 | [DPO 专题][dpo] |
| 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback（RLHF） | 利用人类偏好信号来改进模型行为的一类流程；仓库通过 DPO 讨论相关目标。 | [DPO 专题][dpo] |
| 奖励边际 | reward margin | 偏好回答与非偏好回答所得奖励的差值，用于观察偏好区分程度。 | [DPO 专题][dpo] |
| 合成指令数据 | synthetic instruction data | 由模型生成并筛选的指令与回答样本。 | [数据生成专题][data-gen] |
| 反思式数据精炼 | reflection tuning | 用模型分析并改写现有指令或回答，再提取改进后的训练数据。 | [数据生成专题][reflection] |
| 近似重复检测 | near-duplicate detection | 找出语义或字面上高度相近的样本，避免数据集中出现大量重复项。 | [数据工具][dataset-tools] |
| TF-IDF | Term Frequency–Inverse Document Frequency（TF-IDF） | 衡量词项对文档区分力的文本特征；仓库用于近似重复检测。 | [数据工具][dataset-tools] |
| 自动评估 | automated evaluation | 用代码、参考答案或评估模型对生成结果作系统化检查。 | [评估专题][eval] |
| 模型评估者 | LLM-as-a-judge | 使用另一个语言模型对回答质量进行打分或比较；结果受提示与评估模型影响。 | [评估专题][eval] |
| 基准测试 | benchmark | 在固定任务与评估规则上比较不同模型的能力。 | [第 7 章][ch07] |
| MMLU | Massive Multitask Language Understanding（MMLU） | 涵盖多学科选择题的模型知识能力基准。 | [第 7 章][ch07] |

## 8. 计算、设备与性能

| 名词 / 术语 | 英文 / 缩写 | 简要说明 | 仓库资料 |
| --- | --- | --- | --- |
| 浮点运算次数 | Floating-Point Operations（FLOPs） | 一次计算执行的浮点运算总数，用于估计计算工作量。 | [性能分析][perf] |
| 每秒浮点运算次数 | Floating-Point Operations Per Second（FLOPS） | 浮点计算吞吐率；与表示运算总数的 FLOPs 不是同一量。 | [性能分析][perf] |
| 模型算力利用率 | Model FLOPs Utilization（MFU） | 将实际词元吞吐折算的模型计算量，与硬件理论峰值算力比较的利用率指标。 | [性能分析][perf] |
| 中央处理器 | Central Processing Unit（CPU） | 通用计算设备，可运行模型但大型模型通常更慢。 | [附录 A][appendix-a] |
| 图形处理器 | Graphics Processing Unit（GPU） | 擅长并行张量运算的计算设备，常用于模型训练与推理。 | [附录 A][appendix-a] |
| CUDA | Compute Unified Device Architecture（CUDA） | NVIDIA 的 GPU 计算平台；在 PyTorch 中常作为设备类型。 | [附录 A][appendix-a] |
| MPS | Metal Performance Shaders（MPS） | Apple 设备上供 PyTorch 使用的加速后端。 | [CPU/MPS 差异][cpu-mps] |
| 显存 | Video Random Access Memory（VRAM） | GPU 存放模型参数、激活、梯度与 KV 缓存等数据的内存。 | [KV 缓存专题][kv] |
| 数据类型 / 精度 | data type（dtype） | 张量的数值表示方式，例如 FP32 或 BF16；影响内存和计算特性。 | [GQA 专题][gqa] |
| 单精度浮点 | FP32 / float32 | 每个数通常占 4 字节的浮点格式。 | [GQA 专题][gqa] |
| 半精度浮点 | FP16 / float16 | 每个数通常占 2 字节的浮点格式，数值范围小于 FP32。 | [GQA 专题][gqa] |
| 脑浮点 16 位 | bfloat16（BF16） | 每个数通常占 2 字节、指数范围接近 FP32 的浮点格式。 | [GQA 专题][gqa] |
| 分布式数据并行 | Distributed Data Parallel（DDP） | 多个进程各持有模型副本、处理不同数据并同步梯度的训练方式。 | [训练加速专题][train-speed] |
| 分布式采样器 | DistributedSampler | 将训练数据分配给多个进程，避免各进程反复处理相同样本。 | [附录 A][appendix-ddp] |
| TF32 | TensorFloat-32（TF32） | 在受支持的 NVIDIA GPU 上加速部分 float32 矩阵乘法的计算模式。 | [训练加速专题][train-speed] |
| 编译优化 | `torch.compile` | PyTorch 对模型计算进行编译优化的入口，可能提升运行速度。 | [训练加速专题][train-speed] |
| 内存不足 | Out of Memory（OOM） | 设备无法满足当前内存分配请求；常与批次、序列长度和模型大小有关。 | [训练加速专题][train-speed] |

## 9. 仓库中的模型、数据与工具专名

以下名称有助于定位示例和资料；它们是模型家族、数据集或工具名，不都属于可逐字扩写的技术缩写。

| 名称 | 类别 | 在本仓库中的用途 | 仓库资料 |
| --- | --- | --- | --- |
| GPT-2 | 模型 | 第 5 章加载其预训练权重，并用同类结构演示生成与微调。 | [第 5 章][ch05] |
| Llama | 模型家族 | 扩展材料将基础 GPT 实现逐步转换为 Llama 架构。 | [Llama 专题][llama] |
| Qwen | 模型家族 | 扩展材料实现稠密和 MoE 等模型变体。 | [Qwen 专题][qwen] |
| Gemma | 模型家族 | 扩展材料展示另一组 LLM 架构及 KV 缓存实现。 | [Gemma 专题][gemma] |
| Olmo | 模型家族 | 扩展材料展示 Olmo 3 架构与生成示例。 | [Olmo 专题][olmo] |
| Tiny Aya | 模型家族 | 扩展材料展示并行 Transformer 块等结构。 | [Tiny Aya 专题][tiny-aya] |
| tiktoken | 分词工具 | 用于 GPT 系列的词元编码，并在扩展材料中演示扩充词表。 | [分词器扩展][tokenizer-ext] |
| PyTorch | 机器学习框架 | 承载仓库中的张量、模型、自动求导与训练循环。 | [附录 A][appendix-a] |
| Ollama | 推理工具 | 用于本地运行模型，并参与部分回答评估或数据生成示例。 | [评估专题][eval] |
| IMDb | 数据集 | 影评分类扩展材料使用的文本分类数据。 | [IMDb 扩展][imdb] |
| Project Gutenberg | 文本语料来源 | 第 5 章扩展使用其公开文本演示更长语料的预训练。 | [Gutenberg 扩展][gutenberg] |

## 缩写查找

| 缩写 | 英文全称 | 中文名称 |
| --- | --- | --- |
| BF16 | Brain Floating Point（16-bit） | 脑浮点 16 位 |
| BOS | Beginning of Sequence | 序列开始标记 |
| BPE | Byte Pair Encoding | 字节对编码 |
| CPU | Central Processing Unit | 中央处理器 |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 计算平台 |
| DDP | Distributed Data Parallel | 分布式数据并行 |
| DPO | Direct Preference Optimization | 直接偏好优化 |
| DSA | DeepSeek Sparse Attention | DeepSeek 稀疏注意力 |
| EOS | End of Sequence | 序列结束标记 |
| FFN | Feed-Forward Network | 前馈网络 |
| FLOPs | Floating-Point Operations | 浮点运算次数 |
| FLOPS | Floating-Point Operations Per Second | 每秒浮点运算次数 |
| FP16 / FP32 | 16-bit / 32-bit Floating Point | 半精度 / 单精度浮点 |
| GaLore | Gradient Low-Rank Projection | 梯度低秩投影 |
| GELU | Gaussian Error Linear Unit | 高斯误差线性单元 |
| GLU | Gated Linear Unit | 门控线性单元 |
| GPT | Generative Pre-trained Transformer | 生成式预训练 Transformer |
| GPU | Graphics Processing Unit | 图形处理器 |
| GQA | Grouped-Query Attention | 分组查询注意力 |
| KDA | Kimi Delta Attention | Kimi Delta 注意力 |
| KV | Key–Value | 键与值 |
| LLM | Large Language Model | 大语言模型 |
| LM | Language Model | 语言模型 |
| LoRA | Low-Rank Adaptation | 低秩自适应 |
| LR | Learning Rate | 学习率 |
| MFU | Model FLOPs Utilization | 模型算力利用率 |
| MHA | Multi-Head Attention | 多头注意力 |
| MLA | Multi-Head Latent Attention | 多头潜在注意力 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| MMLU | Massive Multitask Language Understanding | 多学科语言理解基准 |
| MoE | Mixture of Experts | 混合专家 |
| MPS | Metal Performance Shaders | Apple Metal 加速后端 |
| MQA | Multi-Query Attention | 多查询注意力 |
| NLP | Natural Language Processing | 自然语言处理 |
| OOM | Out of Memory | 内存不足 |
| PAD | Padding | 填充标记 |
| PEFT | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| PPL | Perplexity | 困惑度 |
| ReLU | Rectified Linear Unit | 修正线性单元 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习 |
| RMSNorm | Root Mean Square Normalization | 均方根归一化 |
| RoPE | Rotary Position Embedding | 旋转位置嵌入 |
| SDPA | Scaled Dot-Product Attention | 缩放点积注意力 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| SiLU | Sigmoid Linear Unit | Sigmoid 线性单元 |
| SWA | Sliding Window Attention | 滑动窗口注意力 |
| SwiGLU | Swish-Gated Linear Unit | Swish 门控线性单元 |
| TF32 | TensorFloat-32 | TensorFloat-32 |
| UNK | Unknown | 未知词元 |
| VRAM | Video Random Access Memory | 显存 |

<!-- 上表中的资料链接定义 -->

[repo]: README.md
[ch02]: ch02/01_main-chapter-code/ch02.ipynb
[bpe]: ch02/05_bpe-from-scratch/README.md
[ch03]: ch03/01_main-chapter-code/ch03.ipynb
[efficient-attn]: ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
[buffers]: ch03/03_understanding-buffers/understanding-buffers.ipynb
[ch04]: ch04/01_main-chapter-code/ch04.ipynb
[perf]: ch04/02_performance-analysis/flops-analysis.ipynb
[kv]: ch04/03_kv-cache/README.md
[gqa]: ch04/04_gqa/README.md
[mla]: ch04/05_mla/README.md
[swa]: ch04/06_swa/README.md
[moe]: ch04/07_moe/README.md
[deltanet]: ch04/08_deltanet/README.md
[dsa]: ch04/09_dsa/README.md
[crosskv]: ch04/10_kv-sharing/README.md
[ch05]: ch05/01_main-chapter-code/ch05.ipynb
[scheduler]: ch05/04_learning_rate_schedulers/README.md
[gutenberg]: ch05/03_bonus_pretraining_on_gutenberg/README.md
[llama]: ch05/07_gpt_to_llama/README.md
[mmap]: ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb
[train-speed]: ch05/10_llm-training-speed/README.md
[qwen]: ch05/11_qwen3/README.md
[gemma]: ch05/12_gemma3/README.md
[olmo]: ch05/13_olmo3/README.md
[tiny-aya]: ch05/15_tiny-aya/README.md
[qwen35]: ch05/16_qwen3.5/README.md
[tokenizer-ext]: ch05/09_extending-tokenizers/README.md
[muon]: ch05/18_muon/README.md
[cpu-mps]: ch05/19_cpu_mps_differences/README.md
[ch06]: ch06/01_main-chapter-code/ch06.ipynb
[imdb]: ch06/03_bonus_imdb-classification/README.md
[ch07]: ch07/01_main-chapter-code/ch07.ipynb
[ch07ex]: ch07/01_main-chapter-code/exercise-solutions.ipynb
[dataset-tools]: ch07/02_dataset-utilities/find-near-duplicates.py
[eval]: ch07/03_model-evaluation/README.md
[dpo]: ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
[data-gen]: ch07/05_dataset-generation/llama3-ollama.ipynb
[reflection]: ch07/05_dataset-generation/reflection-gpt4.ipynb
[appendix-a]: appendix-A/01_main-chapter-code/code-part1.ipynb
[appendix-ddp]: appendix-A/01_main-chapter-code/DDP-script.py
[appendix-d]: appendix-D/01_main-chapter-code/appendix-D.ipynb
[lora]: appendix-E/01_main-chapter-code/appendix-E.ipynb
