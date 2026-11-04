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
