# 数学计算过程拆解：Token Embedding (self.tok_emb)

本项目用于详细拆解大语言模型中各类核心算子的底层数学计算与代码实现。

---

## 1. 概念与输入

### 输入数据
- **Token 索引 (`input_ids`)**：
  $$\text{input\_ids} = [[2, 3, 4, 5]], \quad \text{Shape: } (B, T) = (1, 4)$$
  - $B = 1$（批大小 batch size）
  - $T = 4$（序列长度 seq len）

### 权重矩阵
- **词表嵌入权重矩阵 ($W$)**：
  $$W \in \mathbb{R}^{V \times d} = \mathbb{R}^{6 \times 4}$$
  - $V = 6$（词表大小 vocab size，索引为 $0 \sim 5$）
  - $d = 4$（嵌入向量维度 embedding dim）

$$
W = \begin{bmatrix}
 0.3374 & -0.1778 & -0.3035 & -0.5880 \\
 0.3486 &  0.6603 & -0.2196 & -0.3792 \\
-0.1606 & -0.4015 &  0.6957 & -1.8061 \\
 1.8960 & -0.1750 &  1.3689 & -1.6033 \\
-0.7849 & -1.4096 & -0.4076 &  0.7953 \\
 0.9985 &  0.2212 &  1.8319 & -0.3378
\end{bmatrix}
$$

---

## 2. 两种等价视角的数学拆解

### 视角一：查表 / 索引选择 (Lookup Table)
在工程实现（如 PyTorch 的 `nn.Embedding`）中，为了极高的执行效率，直接按索引切片取行：
$$y_{b, t} = W[\text{input\_ids}[b, t], :]$$

- 索引 $2 \rightarrow W[2] = [-0.1606, -0.4015,  0.6957, -1.8061]$
- 索引 $3 \rightarrow W[3] = [ 1.8960, -0.1750,  1.3689, -1.6033]$
- 索引 $4 \rightarrow W[4] = [-0.7849, -1.4096, -0.4076,  0.7953]$
- 索引 $5 \rightarrow W[5] = [ 0.9985,  0.2212,  1.8319, -0.3378]$

最终输出形状为 $(1, 4, 4)$。

### 视角二：线性代数矩阵乘法 (One-Hot 矩阵乘法)
在严格数学意义上，Token 嵌入其实是一个**无偏置项的线性变换层 (Linear Layer)**：

#### 1. 构造 One-Hot 矩阵
对于输入序列 $[2, 3, 4, 5]$，在词表大小 $V = 6$ 的基底空间下，将每个 Token ID 映射为一个行标准单位基向量 $\mathbf{e}_i$：

$$
X_{\text{one-hot}} = \begin{bmatrix}
\mathbf{e}_2 \\
\mathbf{e}_3 \\
\mathbf{e}_4 \\
\mathbf{e}_5
\end{bmatrix} =
\begin{bmatrix}
0 & 0 & \mathbf{1} & 0 & 0 & 0 \\
0 & 0 & 0 & \mathbf{1} & 0 & 0 \\
0 & 0 & 0 & 0 & \mathbf{1} & 0 \\
0 & 0 & 0 & 0 & 0 & \mathbf{1}
\end{bmatrix} \in \mathbb{R}^{4 \times 6}
$$

#### 2. 矩阵乘法完整展开
通过矩阵乘法 $Y = X_{\text{one-hot}} \cdot W$ 进行投影：

$$
Y = \begin{bmatrix}
0 & 0 & \mathbf{1} & 0 & 0 & 0 \\
0 & 0 & 0 & \mathbf{1} & 0 & 0 \\
0 & 0 & 0 & 0 & \mathbf{1} & 0 \\
0 & 0 & 0 & 0 & 0 & \mathbf{1}
\end{bmatrix}
\begin{bmatrix}
 0.3374 & -0.1778 & -0.3035 & -0.5880 \\
 0.3486 &  0.6603 & -0.2196 & -0.3792 \\
-0.1606 & -0.4015 &  0.6957 & -1.8061 \\
 1.8960 & -0.1750 &  1.3689 & -1.6033 \\
-0.7849 & -1.4096 & -0.4076 &  0.7953 \\
 0.9985 &  0.2212 &  1.8319 & -0.3378
\end{bmatrix}
$$

#### 3. 逐行计算细节拆解
以第一个位置 $t=0$（Token ID = 2）为例，该行的输出向量为 One-Hot 行向量与权重矩阵的内积求和：

$$
\begin{aligned}
\mathbf{y}_0 &= \mathbf{e}_2 \cdot W \\
&= 0 \cdot \mathbf{w}_0 + 0 \cdot \mathbf{w}_1 + \mathbf{1} \cdot \mathbf{w}_2 + 0 \cdot \mathbf{w}_3 + 0 \cdot \mathbf{w}_4 + 0 \cdot \mathbf{w}_5 \\
&= \mathbf{w}_2 \\
&= [-0.1606, -0.4015,  0.6957, -1.8061]
\end{aligned}
$$

同理可得所有位置的计算结果：

$$
Y = \begin{bmatrix}
\mathbf{w}_2 \\
\mathbf{w}_3 \\
\mathbf{w}_4 \\
\mathbf{w}_5
\end{bmatrix} =
\begin{bmatrix}
-0.1606 & -0.4015 &  0.6957 & -1.8061 \\
 1.8960 & -0.1750 &  1.3689 & -1.6033 \\
-0.7849 & -1.4096 & -0.4076 &  0.7953 \\
 0.9985 &  0.2212 &  1.8319 & -0.3378
\end{bmatrix}
$$

维度变换过程为：
$$(B, T, V) \times (V, d) = (1, 4, 6) \times (6, 4) \longrightarrow (1, 4, 4)$$

---

## 3. 数学本质与工程实现对比

| 视角 | 计算形式 | 时间与空间复杂度 | 优缺点 |
| :--- | :--- | :--- | :--- |
| **数学线性代数视角** | $Y = X_{\text{one-hot}} \cdot W$ | 时间: $O(B \cdot T \cdot V \cdot d)$<br>空间: 需生成 $(B, T, V)$ 稀疏张量 | 数学形式统一严谨，但存在海量冗余的 $0 \times w$ 运算和内存占用 |
| **工程实现视角 (`nn.Embedding`)** | $Y = W[\text{input\_ids}]$ | 时间: $O(B \cdot T \cdot d)$<br>空间: 直接按内存偏移切片 | 避开稀疏矩阵相乘，直接指针寻址提取对应行向量，速度提升成百上千倍 |

---

## 4. 运行代码
运行拆解脚本：
```bash
python gpt2toy/math_breakdown/token_embedding_step_by_step.py
```
