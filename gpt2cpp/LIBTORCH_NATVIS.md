# Visual Studio 中的 LibTorch 对象可视化：Natvis 实现与使用

本项目使用 [`debug/libtorch.natvis`](debug/libtorch.natvis)，让 Visual Studio 的“局部变量”“监视”和悬停提示直接显示张量的形状、步长、数据类型、设备以及部分 CPU 数值，并简化 `Linear`、`Sequential` 等模块的展开。

适用范围为 Windows x64、MSVC STL、Visual Studio 2022。2026-10-28 已核对本机 LibTorch 2.10.0+cpu 与 Python PyTorch 2.14.1+cpu，并分别完成微软原生 C++ 调试器实测。文件依赖库内部字段，升级 LibTorch 或 MSVC 后需要重新检查。

## 1. 为什么默认对象展开难以使用

`torch::Tensor` 是张量句柄，数据并不直接作为一个 C++ 数组存放在它的成员中。常见的访问关系是：

```text
at::Tensor / at::TensorBase
  └─ impl_.target_ → c10::TensorImpl
       ├─ sizes_and_strides_       形状与步长
       ├─ storage_offset_         视图在存储中的起点，单位为元素
       ├─ data_type_              数据类型
       ├─ device_opt_             设备
       └─ storage_.storage_impl_.target_
            └─ data_ptr_.ptr_.data_ → 底层存储
```

模块也有包装层。例如 `torch::nn::Linear` 继承 `ModuleHolder<LinearImpl>`，通过共享指针持有实现对象，`weight`、`bias` 和选项保存在实现对象中。默认调试视图会同时暴露这些包装、继承与指针结构。[PyTorch C++ Frontend 官方教程](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html)介绍了这种模块存储方式。

Natvis 用 XML 规定这些类型的摘要和展开内容。普通表达式可以读取字段、执行算术和指针访问；自定义 `Intrinsic` 可以封装调试器中的表达式。本文件没有调用被调试进程的成员函数。[Microsoft Natvis 文档](https://learn.microsoft.com/en-us/visualstudio/debugger/create-custom-views-of-native-objects?view=vs-2022)介绍了规则、加载方式和表达式限制。

本次还检查了 PyTorch 官方主仓库的[文件树](https://api.github.com/repos/pytorch/pytorch/git/trees/36d79410c724c707af4263afb9b4de33fc730517?recursive=1)，以及[官方 Windows CPU 发布目录](https://download.pytorch.org/libtorch/cpu/)中的 `libtorch-win-shared-with-deps-2.13.0+cpu.zip` 文件目录，均未发现单独的 `.natvis`。这个检查不覆盖所有版本、CUDA 包或 PDB 内嵌信息。官方仓库有面向 GDB 的 [`pytorch-gdb.py`](https://github.com/pytorch/pytorch/blob/main/tools/gdb/pytorch-gdb.py)；官方文档提到的 [VS 扩展](https://docs.pytorch.org/cppdocs/installing.html#visual-studio-extension)用于配置项目与链接选项。

## 2. 在项目中使用

[`CMakeLists.txt`](CMakeLists.txt) 已在 MSVC 分支加入：

```cmake
target_sources(gpt PRIVATE debug/libtorch.natvis)
source_group("Debugger" FILES debug/libtorch.natvis)
```

对已有构建目录重新运行原来的 CMake 配置命令，保持已选择的 Python 和 LibTorch 路径。在 VS 中重新加载生成的 `gpt` 项目，再启动调试。VS 可以从项目加载 Natvis，构建时也会将项目中的 Natvis 纳入生成的 PDB。

在张量初始化完成后设置断点，例如：

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor x = torch::arange(6, torch::kFloat32).reshape({2, 3});
    torch::Tensor transposed = x.transpose(0, 1);
    torch::nn::Linear linear(3, 2);
    torch::Tensor y = linear->forward(x);

    // 在这一行设置断点，前面的变量均已初始化。
    std::cout << y << '\n';
}
```

展开 `x` 时可看到：

```text
Tensor shape=[2, 3], dtype=float32, device=cpu, numel=6
  shape           [2, 3]
  strides         [3, 1]
  dtype           float32
  device          cpu
  numel           6
  storage_offset  0
  contiguous      true
  [values]        First 6 of 6 elements (logical row-major order)
    [0]           0
    [1]           1
    [2]           2
    [3]           3
    [4]           4
    [5]           5
```

其他 VS C++ 项目可以通过“添加 → 现有项”加入同一份 `.natvis`。需要对多个项目生效时，也可以复制到 `%USERPROFILE%\Documents\Visual Studio 2022\Visualizers`；应避免项目、用户目录和 PDB 中同时保留版本不同的同名规则。

使用此文件不需要安装 VS Code。第 6 节的自动验证使用 VS Code 作为微软原生调试器的测试宿主。

## 3. 当前支持范围

| 对象或情况 | 显示行为 |
| --- | --- |
| 普通张量 | 形状、步长、dtype、device、numel、存储偏移、连续性 |
| CPU 基础类型 | 显示前 64 个逻辑元素，支持 `float32`、`float64`、`int64`、`int32`、`int16`、`int8`、`uint8`、`bool` |
| 转置、切片、广播 | 结合步长与存储偏移计算数值位置 |
| 零维标量 | 形状为 `[]`，预览一个元素 |
| 空张量 | 显示空预览，不读取数据 |
| 未定义张量 | 显示 `Tensor (undefined)` |
| 超过 5 维 | 摘要缩略，展开 `shape` / `strides` 查看全部维度；数值预览最多处理 64 维 |
| CUDA、Meta | 保留可用元数据，不读取数值 |
| float16、bfloat16、复数、量化及其他未支持 dtype | 保留元数据，不解释底层数值 |
| 特殊布局、符号或自定义元数据 | 不把缓存形状当成真实形状，显示对应提示 |
| 惰性共轭、惰性取负视图 | 不直接把尚未变换的存储值当作逻辑值展示 |
| `ModuleHolder` | 穿过包装层展开实现对象，并识别空模块 |
| `LinearImpl` | 显示输入/输出维度、偏置选项、权重和偏置 |
| `SequentialImpl` | 显示层列表，通过 `AnyModule` 继续展开实际模块 |

`[values]` 是受限的扁平预览，顺序与逻辑行主序一致。它不会改变张量、分配副本、执行 `.cpu()` 或发起 CUDA 同步。`[Raw View]` 仍可用于检查原始字段。`autograd` 节点在存在元数据且表达式可用时显示原始指针；当前没有专门格式化梯度或计算图。

## 4. 形状和数值如何解析

### 4.1 内联与外置的形状存储

当前两版 `c10::impl::SizesAndStrides` 在维度不超过 5 时使用内联数组：前 5 个槽位预留给形状，后 5 个槽位预留给步长。因此二维张量的步长起点仍是第 5 个槽位，不能按实际维度数定位。

维度超过 5 时，形状和步长存放在外置数组中：前 `rank` 项是形状，随后 `rank` 项是步长。Natvis 中的 `sizes_ptr()`、`strides_ptr()` 和索引辅助表达式分别处理两种情况。实现依据为本机安装包中的 `c10/core/impl/SizesAndStrides.h`。

未定义张量也不能仅通过 `impl_` 是否为空判断：句柄可能指向未定义张量单例。当前规则结合 `device_opt_._Has_value` 识别该状态，并在展开内部结构之前处理。

### 4.2 逻辑索引到存储索引

读取视图时，底层存储地址和逻辑张量的起点可能不同。规则将扁平逻辑索引逐维拆解，再使用步长计算存储位置：

```text
remaining = flat_index
offset = storage_offset
从最后一个维度向前遍历：
    coordinate = remaining % shape[dim]
    remaining = remaining / shape[dim]
    offset += coordinate * stride[dim]
value = typed_storage[offset]
```

例如 `x` 的数据为 `[[0, 1, 2], [3, 4, 5]]`：

| 对象 | shape | stride | storage_offset | 预览顺序 |
| --- | --- | --- | --- | --- |
| `x` | `[2, 3]` | `[3, 1]` | `0` | `0, 1, 2, 3, 4, 5` |
| `x.transpose(0, 1)` | `[3, 2]` | `[1, 3]` | `0` | `0, 3, 1, 4, 2, 5` |
| `x.slice(0, 1, 2).slice(1, 1, 3)` | `[1, 2]` | `[3, 1]` | `4` | `4, 5` |

广播维度的步长可以为 0，上述计算会重复读取同一存储位置。零维标量没有维度循环，直接使用存储偏移。空张量跳过整个读取过程，避免对大小为 0 的维度做除法。

`storage_offset_` 的单位为元素。规则先按 dtype 转换指针类型，再使用元素偏移，避免把 `int64`、`bool` 等类型错误地解释为 `float`。

### 4.3 哪些情况禁止读取数值

数值预览要求张量和底层存储都属于 CPU，元数据使用普通布局，存储可访问，dtype 已支持，并且没有惰性共轭或取负标记。检查失败时显示 `[values unavailable]`。

这些限制用于保证显示含义正确。例如，转置只是改变步长，可以按索引解析；共轭或取负视图还会改变元素的逻辑数值，直接读取存储会显示错误结果。GPU 地址也不能交给普通 CPU 指针预览。

所有 `Intrinsic` 都是字段读取或算术表达式，`CustomListItems` 中的 `Exec` 仅更新调试器遍历变量，不向被调试对象赋值。

## 5. 常见问题

### 仍然显示大量内部字段

先确认 VS 已重新加载包含 Natvis 的项目，文件未被忽略或移除。打开“工具 → 选项 → 调试”下的输出窗口设置，将 Natvis 诊断消息设为错误或详细级别，然后查看“输出”窗口。

在同一个 VS 实例中编辑并保存 Natvis 后，变量窗口可以刷新。从外部修改文件后，可以在“即时”窗口执行 `.natvisreload`，或重新开始调试。若规则不匹配当前库的字段，VS 会回退到原始展开，应根据诊断信息核对实际头文件。

### 变量被优化掉，或者类型信息不可用

Natvis 只能解释调试器能读取的对象，不能恢复已被编译器消除的局部变量。应用需要匹配的 PDB；排查时可以在保持 `/MD` 的前提下，使用带调试信息、关闭优化的应用构建。本次验证使用 `RelWithDebInfo` 并覆盖为 `/Od`。

查看常用张量字段不一定需要完整的 LibTorch DLL 调试符号；本次测试依靠应用 PDB 中的类型信息完成了展开。继续深入库函数、动态内部类型或 autograd 对象时，可能还需要对应模块的匹配符号。

Release LibTorch 与使用 Debug CRT 的默认 Debug 应用不能混用。PyTorch 已公告从 2.12.1 起停止发布 Windows 预编译 LibTorch Debug 包；确需 Debug 库时参考[官方源码构建公告](https://dev-discuss.pytorch.org/t/important-note-stop-publishing-prebuilt-windows-libtorch-debug-binaries/3400)。本项目的构建设置见[编译与运行说明](BUILD_INSTRUCTIONS.md)。

### 预览数量有限，或者提示数值不可用

默认只展示前 64 项，以控制读取量并适应调试器的集合展示限制。需要观察其他位置时，可以在程序中显式准备所需的小切片或 CPU 张量，再观察该变量。不要在 Natvis 中自动搬运整张 GPU 张量。

对于未支持 dtype、布局或惰性视图，可先通过正常程序执行路径准备便于查看的张量。查看之前需留意转换和拷贝会产生的开销，以及新张量是否仍与原存储共享内存。

## 6. 验证记录与边界

验证日期为 2026-10-28，环境为 Windows x64、VS 2022 / MSVC 19.44、C++20、`RelWithDebInfo`、`/MD`、`/Od`。测试工程将 Natvis 加入项目和链接设置，生成 exe/PDB，再通过正式 Microsoft VS Code 宿主中的原生 C++ 调试器，在断点处调用 `evaluate` / `variables` 检查真实展开。

| 库 | 编译、链接和运行 | 调试器检查 |
| --- | --- | --- |
| 独立 LibTorch 2.10.0+cpu | 通过 | 28 项通过 |
| Python PyTorch 2.14.1+cpu 附带的 C++ 库 | 通过 | 28 项通过 |

覆盖范围包括八种基础 dtype 的名称和数值、标量、空与未定义张量、转置、非零偏移切片、6 维形状与步长、零步长广播、64 项预览上限、特殊张量的数值屏蔽，以及空/非空 `ModuleHolder`、`LinearImpl` 和 `SequentialImpl`。其中转置、切片、广播的预览与预期逻辑顺序逐项比较，不能只凭能展开对象就判定通过。

同时通过 VS 2022 自带 `natvis.xsd` 格式校验，并确认重新生成后的 `gpt.vcxproj` 包含 Natvis 项。没有通过自动化直接检查 VS 2022 的“监视/局部变量”GUI；CPU 环境未覆盖 CUDA 实机。2.14.1 配置有 `kineto` 库缺失警告，但本次测试程序编译、链接和运行成功。

重新验证时，可以从第 2 节的小程序开始，再加入第 4 节的转置、切片和广播案例。检查 shape、stride、offset 与数值是否同时正确，并确认 Meta、稀疏和惰性视图不会出现未经变换的原始数值预览。

## 7. Git 中保留与排除的内容

仓库保留 Natvis 文件、CMake 接入和本技术文档。构建目录、VS 缓存以及本次使用的本机验证辅助工程由根目录 `.gitignore` 排除：

```gitignore
/gpt2cpp/build/
/gpt2cpp/build-*/
/gpt2cpp/.vs/
/gpt2cpp/debug/validation/
```

本机验证辅助工程保留在 `gpt2cpp/debug/validation/`，不作为仓库依赖。它的二进制、PDB、独立 VS Code 配置、扩展引用和详细日志写入 `%TEMP%` 下的专用构建目录。验证结果已汇总在第 6 节，文档不依赖这些本地文件才能阅读或使用 Natvis。
