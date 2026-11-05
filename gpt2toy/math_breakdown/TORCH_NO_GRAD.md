# `with torch.no_grad()`：关闭梯度记录

`torch.no_grad()` 创建一个上下文管理器；在 `with` 代码块中，PyTorch 通常不为运算记录反向传播所需的计算图。它常用于推理，可以减少内存开销，有时也能提高速度。

配套的可运行示例：[torch_no_grad_example.py](torch_no_grad_example.py)。安装 PyTorch 后，在本目录运行：

```bash
python torch_no_grad_example.py
```

## 基本用法

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3

z = x * 4
z.backward()

print(y.item(), y.requires_grad)  # 6.0 False
print(z.item(), z.requires_grad)  # 8.0 True
print(x.grad.item())             # 4.0
```

`y` 的乘法发生在 `no_grad` 中，因此没有记录从 `x` 到 `y` 的运算，不能通过 `y.backward()` 求导。`z` 的乘法发生在退出之后，假设进入前梯度记录是开启的，它可以正常反向传播。退出后不会为 `y` 补建计算图。

## `with` 如何恢复状态？

`torch.no_grad` 是 PyTorch 定义的类，`torch.no_grad()` 创建其实例。这个实例实现了上下文管理器方法，下面是帮助理解的简化示意：

```python
class SimplifiedNoGrad:
    def __enter__(self):
        self.previous_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.previous_state)
```

Python 在进入 `with` 时调用 `__enter__()`，退出时调用 `__exit__()`。即使代码块抛出异常，也会调用退出方法来恢复状态。恢复的是**进入前的状态**，并非总是开启；所以嵌套 `no_grad` 时，退出内层仍处于关闭状态。梯度模式是线程局部的，不会自动改变其他线程的状态。

## `torch.is_grad_enabled()` 如何获取线程状态？

`torch.is_grad_enabled()` 查询当前线程是否开启**反向模式**的梯度记录，返回 `True` 或 `False`。它查询的是线程的模式，不是某个张量的 `requires_grad`。张量运算是否记录计算图还取决于输入等其他条件。

在当前环境的 PyTorch `2.14.1+cpu` 中，它是一个 C++ 扩展提供的内置函数，因此 `inspect.getsource(torch.is_grad_enabled)` 无法像读取普通 Python 函数那样显示实现。对应的源码版本可以用 `torch.version.git_version` 查看。下面以当前版本的源码为例，展示主要调用链：

```text
torch.is_grad_enabled()                     Python 接口
    ↓
is_grad_enabled()                           C++ 扩展入口
    ↓
GradMode::is_enabled()
    ↓
AutogradState::get_tls_state().get_grad_mode()
    ↓
当前线程的 AutogradState.grad_mode_
```

### C++ 扩展入口

[autograd/init.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/init.cpp) 的实现中，入口调用 `GradMode::is_enabled()`，根据结果返回 Python 布尔值；以下摘录省略了异常处理宏：

```cpp
if (GradMode::is_enabled()) {
  Py_RETURN_TRUE;
} else {
  Py_RETURN_FALSE;
}
```

### 获取线程局部状态

[GradMode.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/c10/core/GradMode.cpp) 中，读取操作获取当前线程的状态对象，再读取其梯度模式：

```cpp
bool GradMode::is_enabled() {
  return AutogradState::get_tls_state().get_grad_mode();
}
```

[AutogradState.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/c10/core/AutogradState.cpp) 用 C++ 的 `thread_local` 声明状态对象，读取函数返回该对象：

```cpp
thread_local AutogradState autograd_state_tls = AutogradState(
    /* grad_mode */ true,
    /* inference_mode */ false,
    /* fw_grad_mode */ true,
    /* multithreading_enabled */ true);

AutogradState& AutogradState::get_tls_state() {
  return autograd_state_tls;
}
```

同一个变量名在每个线程中对应各自的状态。`get_grad_mode()` 返回状态对象的 `grad_mode_` 字段；`no_grad` 通过设置当前线程的梯度模式来改变它，并在退出时恢复之前的值。具体代码以所安装版本为准。

### 多线程示例

```python
import threading
import torch


def worker():
    print("子线程:", torch.is_grad_enabled())


print("主线程进入前:", torch.is_grad_enabled())
with torch.no_grad():
    print("主线程 no_grad 内:", torch.is_grad_enabled())
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()  # 等子线程打印完成

print("主线程退出后:", torch.is_grad_enabled())
```

在新启动的普通 Python 进程中，通常输出：

```text
主线程进入前: True
主线程 no_grad 内: False
子线程: True
主线程退出后: True
```

主线程的 `no_grad` 不会改变子线程的梯度模式；退出后主线程恢复为进入前的值。如果主线程进入前已关闭梯度记录，退出后也会保持关闭。

## 模型推理

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

`model.eval()` 切换模型中 Dropout、BatchNorm 等层的行为；`torch.no_grad()` 控制梯度记录。两者作用不同，推理时通常一起使用。`no_grad` 不会修改模型参数的 `requires_grad` 属性，也不会自动切换模型的训练或评估模式。

## 边界

- `no_grad` 影响反向模式自动微分的记录；它不关闭前向模式自动微分。
- 显式接受 `requires_grad` 参数的工厂函数是例外：例如，在代码块内创建 `torch.nn.Parameter`，默认仍可得到 `requires_grad=True` 的参数。
- 训练中需要通过某段运算反向传播时，不应把那段运算放进 `no_grad`。
