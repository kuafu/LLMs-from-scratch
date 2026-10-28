# register_module：注册子模块

`register_module` 的作用是将一个神经网络模块登记为当前模块的子模块，让父模块能够统一管理它。

- `register`：登记、注册。
- `module`：模块，例如线性层、Dropout 层或整个注意力模块。

本文以项目中的 LibTorch C++ 代码为例，最后对比 Python 的写法。

## 从项目代码看

在 [gpt2cpp/gpt.cpp](../gpt2cpp/gpt.cpp#L102) 的 `MultiHeadAttentionImpl` 构造函数中：

```cpp
W_query = register_module(
    "W_query",
    torch::nn::Linear(
        torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)
    )
);
```

可以将这行代码拆成三步：

```cpp
// 1. 创建一个线性层。
auto layer = torch::nn::Linear(
    torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)
);

// 2. 将线性层以 "W_query" 为名字登记到当前模块。
auto registered_layer = register_module("W_query", layer);

// 3. 保存到成员变量，供后续代码使用。
W_query = registered_layer;
```

这里有两个不同的名字：

- `"W_query"` 是传给 `register_module` 的登记名字，供父模块查找和组织子模块。
- `W_query` 是 C++ 成员变量，方便代码直接访问线性层。

它们通常写成相同的名字，便于对应，但并不要求相同。登记成功后，父模块看到的结构可以理解为：

```text
MultiHeadAttentionImpl
├── W_query  -> Linear
├── W_key    -> Linear
├── W_value  -> Linear
├── out_proj -> Linear
└── dropout  -> Dropout
```

例如，`MultiHeadAttentionImpl` 的 `named_parameters()` 可以找到 `W_query.weight`；若 `qkv_bias` 为 `true`，还可以找到 `W_query.bias`。

### 返回的是什么

对这里的 LibTorch C++ 接口，`register_module` 返回指向已注册模块的 `std::shared_ptr`。`torch::nn::Linear` 是模块的持有包装，成员变量和父模块内部的登记记录会共同持有同一个线性层，而不是创建两套权重。

注册负责登记模块；上面的线性层在执行 `torch::nn::Linear(...)` 时创建。后续前向计算发生在 [forward](../gpt2cpp/gpt.cpp#L112) 中，例如：

```cpp
auto queries = linear_like_python(x, *W_query);
```

## 为什么要注册

注册以后，父模块可以递归找到子模块，并统一执行这些操作：

| 对父模块的操作 | 子模块获得的效果 |
| --- | --- |
| `parameters()` / `named_parameters()` | 收集子模块已注册的参数及其名字 |
| `to(device)` | 移动子模块的参数和缓存 |
| `train()` / `eval()` | 切换子模块模式，例如影响 Dropout |
| 保存和加载模型 | 处理子模块已注册的状态，例如参数和缓存 |

例如，构造优化器时经常会传入父模块的 `parameters()`。通过注册，优化器能够拿到子模块的参数；是否更新还取决于梯度、优化器配置等条件。

如果只把线性层赋给 C++ 成员变量，而没有注册，代码仍然可以调用该层，自动求导也能计算梯度。但父模块的 `parameters()` 不会通过这个未注册的成员找到它的参数。因此，如果优化器只使用父模块的 `parameters()`，它就不会更新该层的参数。

## 与 Python 的区别

项目的 [Python 版本](../ch04/01_main-chapter-code/gpt.py#L64) 写成：

```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
```

对于继承了 `nn.Module` 的类，这样赋值时 PyTorch 会自动将 `W_query` 注册为子模块。构造函数中要先调用 `super().__init__()`，然后再进行这种赋值。

Python 也提供 `register_module(name, module)`，它是 `add_module()` 的别名。它和 C++ 的返回值不同：Python 版本返回 `None`。

```python
import torch.nn as nn

model = nn.Module()
result = model.register_module("linear", nn.Linear(3, 4))

print(result)        # None
print(model.linear)  # 已注册的 Linear 模块
```

因此，不要把 Python `register_module` 的返回值当成模块，再赋给成员变量。

## 相关的注册方法

项目中还使用了另外两种注册方式：

| 方法 | 注册对象 | 项目示例 |
| --- | --- | --- |
| `register_module` | 子模块 | [注册 `W_query` 线性层](../gpt2cpp/gpt.cpp#L102) |
| `register_parameter` | 参数张量 | [注册 LayerNorm 的 `scale`](../gpt2cpp/gpt.cpp#L164) |
| `register_buffer` | 缓存张量 | [注册注意力掩码 `mask`](../gpt2cpp/gpt.cpp#L109) |

例如，`mask` 不是优化器需要更新的参数，但它仍属于模型状态，需要随模型移动设备。

## 参考

- [PyTorch C++ Frontend：模块和注册](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html)
- [PyTorch Python `nn.Module` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
