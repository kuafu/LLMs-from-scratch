# Python 对象调用与源码定位参考

这份参考面向阅读本仓库 PyTorch Notebook 和脚本时常见的 Python 问题：如何定位对象的实现文件、为什么某些对象无法查看 Python 源码、包中的类为什么可以通过 `nn.ReLU` 访问，以及 `activation(out)` 这样的调用语法如何工作。

它不替代 Python 语言教程或 PyTorch API 文档；目标是帮助你用少量检查代码判断“当前变量是什么、它从哪里来、调用时会执行什么”。不同 Python、PyTorch 版本和运行环境的文件路径可能不同，应以本机输出为准。

## 1. 先区分：模块、类、实例、函数和原生算子

同一个名字附近的对象可能处在不同层次。以 ReLU 为例：

```python
import torch
from torch import nn

activation = nn.ReLU()

print(nn.ReLU)          # 类：用于创建 ReLU 层
print(activation)       # 实例：一个具体的 ReLU 层对象
print(torch.relu)       # 原生算子：直接处理张量
```

| 对象 | 作用 | 常见调用方式 |
| --- | --- | --- |
| 模块，例如 `torch` | 组织类、函数和子模块 | `import torch` |
| 类，例如 `nn.ReLU` | 描述如何创建对象 | `nn.ReLU()` |
| 实例，例如 `activation` | 保存对象自身的状态和行为 | `activation(x)` |
| Python 函数，例如 `torch.nn.functional.relu` | 用 Python 定义的可调用对象 | `F.relu(x)` |
| 原生算子，例如 `torch.relu` | 由 PyTorch 底层实现的可调用对象 | `torch.relu(x)` |

先用 `type()` 看对象的实际类别，通常比只看变量名更可靠：

```python
import torch
from torch import nn

activation = nn.ReLU()

print(type(nn.ReLU))
print(type(activation))
print(type(torch.relu))
```

## 2. 定位 Python 模块和源码文件

### 2.1 `__file__`：模块加载自哪个文件

Python 模块通常提供 `__file__` 属性。结合 `pathlib.Path.resolve()` 可以得到绝对路径：

```python
from pathlib import Path
import torch

print(Path(torch.__file__).resolve())
print(Path(torch.__file__).resolve().parent)
```

第一个表达式通常指向 `torch/__init__.py`，第二个指向 `torch` 包目录。`__file__` 适合查询模块；它不保证存在于每一种模块上，例如某些内置模块或交互环境创建的模块。

### 2.2 `inspect.getfile()`：定义文件在哪里

`inspect.getfile()` 可定位 Python 模块、类、方法、普通函数和代码对象的定义文件：

```python
import inspect
from torch import nn
import torch.nn.functional as F

print(inspect.getfile(nn.ReLU))
print(inspect.getfile(F.relu))
```

`inspect.getsource()` 则尝试读取对应的 Python 源代码：

```python
print(inspect.getsource(nn.ReLU))
```

这两个函数适合检查 Python 实现的对象，不是所有可调用对象都适用。

### 2.3 为什么 `inspect.getfile(torch.relu)` 会报错

在许多 PyTorch 安装中，`torch.relu` 的类型是 `builtin_function_or_method`。它是 Python 暴露出来的原生算子，而不是保存在某个 `.py` 文件中的普通 Python 函数：

```python
import inspect
import torch

print(type(torch.relu))
print(inspect.getfile(torch.relu))  # 通常抛出 TypeError
```

这种 `TypeError` 不表示 ReLU 不存在，而是表示 `inspect` 找不到可读取的 Python 源码文件。可以查看 Python 包装函数：

```python
import inspect
import torch.nn.functional as F

print(inspect.getsource(F.relu))
```

也可以定位已安装的原生扩展二进制：

```python
from pathlib import Path
import torch

print(Path(torch._C.__file__).resolve())
```

在 Windows 中该路径通常以 `.pyd` 结尾，在 Linux 或 macOS 中通常以 `.so` 结尾。它是编译后的二进制文件，不是可直接阅读的 C++ 源码；如需研究底层实现，应查看与已安装 PyTorch 版本匹配的上游源码。

### 2.4 包导入：为什么 `activation.py` 中的类可以写成 `nn.ReLU`

这里涉及三件事：目录对应包的层级、导入语句绑定名字，以及包通过 `__init__.py` 逐层导出名字。

#### 目录怎样对应包名

对于安装在文件系统中的普通 Python 包，带有 `__init__.py` 的目录对应一个包，普通 `.py` 文件对应模块。包本身也是模块对象，有自己的命名空间。以下是 PyTorch 的相关目录结构：

```text
site-packages/                 ← 导入搜索起点，不计入包名
└── torch/
    ├── __init__.py            ← torch 包
    └── nn/
        ├── __init__.py        ← torch.nn 包
        └── modules/
            ├── __init__.py    ← torch.nn.modules 包
            └── activation.py ← torch.nn.modules.activation 模块
```

当 `site-packages` 位于 Python 的导入搜索路径 `sys.path` 中时，导入器可以先找到 `torch`，再沿父包的搜索路径查找 `nn`、`modules` 和 `activation`。目录存在本身不会让其中所有文件和类自动成为可访问的名字。[参见 Python 包教程][python-packages]。

> Python 也支持没有 `__init__.py` 的“命名空间包”；这里的 PyTorch 使用普通包形式。包和模块也不一定来自普通目录，例如还可以从 ZIP 文件加载。

#### 为什么在 `__init__.py` 中导入，就成为包的属性

导入普通包时，Python 在这个包的命名空间中执行它的 `__init__.py`。例如，`torch/nn/modules/__init__.py` 的全局命名空间就是 `torch.nn.modules` 的命名空间。

因此，在这个文件的顶层写：

```python
# torch/nn/modules/__init__.py 中与本例有关的导入
from .activation import ReLU
```

会从 `torch.nn.modules.activation` 取得 `ReLU` 类，并在当前包中绑定名字 `ReLU`，于是可以访问 `torch.nn.modules.ReLU`。这里的 `.` 表示当前包，所以 `.activation` 是当前包下的 `activation` 模块。

“把类放进包的命名空间”就是建立这个名称绑定：包对象的 `__dict__["ReLU"]` 引用该类对象。类的定义仍然在 `activation.py` 中。[参见 Python 导入系统][python-import]。

这个规则也适用于普通赋值：如果在该 `__init__.py` 的顶层写 `x = 123`，包加载后就可以通过 `torch.nn.modules.x` 访问它。顶层 `from ... import ...` 同样会建立当前模块或包中的全局名称。

#### `ReLU` 如何继续导出到 `torch.nn`

上面的导入建立了 `torch.nn.modules.ReLU`。在本机 PyTorch 源码中，`torch/nn/modules/__init__.py` 还把 `"ReLU"` 列入 `__all__`；随后 `torch/nn/__init__.py` 执行：

```python
# torch/nn/__init__.py
from torch.nn.modules import *
```

这会把 `__all__` 列出的名字导入当前包，包括 `ReLU`，因此得到 `torch.nn.ReLU`。`__all__` 用于控制星号导入的名称集合；它不会复制类，也不会限制对已有属性的直接访问。未定义 `__all__` 时，星号导入默认取得已有的、不以下划线开头的名字，并不会扫描目录导入全部子模块。[参见星号导入规则][python-import-all]。

最后，在使用代码中写：

```python
import torch.nn as nn
# 也可以写 from torch import nn
```

`nn` 就引用 `torch.nn` 这个包对象，于是 `nn.ReLU` 与 `torch.nn.ReLU` 是同一个类。

完整的名称导出关系为：

```text
torch.nn.modules.activation.ReLU  ← 类的定义位置
    → torch.nn.modules.ReLU       ← modules/__init__.py 中导入
    → torch.nn.ReLU               ← nn/__init__.py 中导入
    → nn.ReLU                     ← 使用者给 torch.nn 起的别名
```

这是同一个类对象的多个访问名称。可以在本机运行以下代码确认导出关系与定义位置：

```python
import inspect
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

print(nn is torch.nn)                          # True
print(nn.modules.__name__)                     # torch.nn.modules
print(nn.modules.__file__)                     # .../torch/nn/modules/__init__.py
print(nn.modules.__dict__["ReLU"] is ReLU)      # True
print(nn.ReLU is ReLU)                         # True
print(nn.ReLU.__module__)                      # torch.nn.modules.activation
print(inspect.getfile(nn.ReLU))                # .../torch/nn/modules/activation.py
```

## 3. `__call__`：对象为什么可以像函数一样调用

在 Python 中，圆括号调用语法不只用于普通函数。只要类定义了 `__call__` 方法，它创建出的实例就可以写成 `对象(参数)`。

```python
class Scale:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


double = Scale(2)
print(double(3))  # 6
```

`double(3)` 大致相当于：

```python
type(double).__call__(double, 3)
```

可以使用 `callable()` 检查对象是否能使用圆括号调用：

```python
print(callable(double))
print(callable(3))
```

### 3.1 两对圆括号分别在做什么

下面的写法容易混淆：

```python
nn.ReLU()(x)
```

它等价于：

```python
activation = nn.ReLU()  # 第一对圆括号：创建 ReLU 实例
result = activation(x)  # 第二对圆括号：调用该实例
```

为了在 Notebook 中观察中间对象或复用同一层，通常建议使用展开后的两行。

`nn.ReLU(out)` 则不是把 `out` 传给 ReLU 计算。这里的圆括号仍在创建 `ReLU` 实例，`out` 会被误当成构造参数 `inplace`；应写成 `nn.ReLU()(out)`、`activation(out)` 或 `torch.relu(out)`。

## 4. PyTorch 的 `Module`、`__call__` 与 `forward`

`nn.ReLU` 没有必要在自己的类定义中重复实现 `__call__`，因为它继承自 `nn.Module`。可用方法解析顺序（MRO）查看继承链：

```python
from torch import nn

for cls in nn.ReLU.mro():
    print(cls)
```

对一个 ReLU 实例，主要调用关系可以理解为：

```text
activation(x)
    → nn.Module.__call__(...)
    → ReLU.forward(x)
    → torch.nn.functional.relu(x)
    → torch.relu(x)
```

其中 `forward()` 定义该层的计算规则。ReLU 的规则是逐元素计算 `max(0, x)`。`__call__` 则负责让模块实例可以写成 `activation(x)`，并在实际框架中处理前向钩子、反向钩子等模块机制。

因此，虽然下列两种写法在简单 ReLU 上会得到相同数值：

```python
activation(x)
activation.forward(x)
```

日常代码应使用 `activation(x)`。PyTorch 官方 API 也建议实现 `forward()`，但通过模块实例调用它。

> `Module.__call__` 内部的私有辅助方法名称会随 PyTorch 版本变化；排查时优先关注稳定关系：模块实例调用会进入该实例的 `forward()`。

## 5. 一套可复用的对象检查模板

遇到陌生对象时，可以按下面的顺序检查：

```python
from pathlib import Path
import inspect
import torch
from torch import nn


def describe(obj):
    print("repr:", repr(obj))
    print("type:", type(obj))
    print("callable:", callable(obj))
    print("module:", getattr(obj, "__module__", None))

    try:
        print("file:", Path(inspect.getfile(obj)).resolve())
    except TypeError:
        print("file: no Python source file available")


describe(nn.ReLU)
describe(nn.ReLU())
describe(torch.relu)
```

如果对象是 Python 函数或类，还可以继续执行：

```python
print(inspect.signature(nn.ReLU))
print(inspect.getsource(nn.ReLU))
help(nn.ReLU)
```

在 Jupyter Notebook 中，以下写法也很实用：

```python
nn.ReLU?
nn.ReLU??
```

单个问号显示文档和基本信息；双问号会在可用时额外显示源码。对于原生算子，依然可能没有 Python 源码可显示。

## 6. 常见问题

| 现象 | 原因 | 建议 |
| --- | --- | --- |
| `inspect.getfile(torch.relu)` 抛出 `TypeError` | `torch.relu` 是原生内置算子，没有对应的 Python 源文件 | 查看 `F.relu` 的 Python 包装，或定位 `torch._C` 二进制；不要把二进制路径当作 C++ 源码路径 |
| `activation(out)` 看起来像函数调用 | `activation` 是实现或继承了 `__call__` 的对象 | 用 `type(activation)`、`callable(activation)` 和 `.mro()` 检查 |
| `nn.ReLU(out)` 不能得到激活结果 | `nn.ReLU(...)` 是构造实例，不是前向计算 | 使用 `nn.ReLU()(out)`、`activation(out)` 或 `torch.relu(out)` |
| `activation.forward(out)` 可以运行 | `forward()` 是层的计算定义 | 常规代码仍使用 `activation(out)`，让 `nn.Module.__call__` 处理框架功能 |
| 路径与他人或示例不同 | Python、PyTorch、虚拟环境或系统不同 | 使用 `Path(...).resolve()` 获取当前环境的实际路径 |

## 7. 官方资料

- [Python 数据模型：`__call__`][python-call]
- [Python `inspect` 模块：`getfile`][python-inspect]
- [Python 教程：包与相对导入][python-packages]
- [Python 导入系统：普通包、命名空间与加载][python-import]
- [Python 教程：从包中进行星号导入][python-import-all]
- [PyTorch `torch.nn.Module`][torch-module]
- [PyTorch Modules 使用说明][torch-modules-note]
- [PyTorch `nn.ReLU`][torch-relu]

[python-call]: https://docs.python.org/3/reference/datamodel.html#object.__call__
[python-inspect]: https://docs.python.org/3/library/inspect.html#inspect.getfile
[python-packages]: https://docs.python.org/3.13/tutorial/modules.html#packages
[python-import]: https://docs.python.org/3.13/reference/import.html#regular-packages
[python-import-all]: https://docs.python.org/3.13/tutorial/modules.html#importing-from-a-package
[torch-module]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
[torch-modules-note]: https://docs.pytorch.org/docs/stable/notes/modules.html
[torch-relu]: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
