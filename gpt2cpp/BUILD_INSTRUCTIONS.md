# GPT-2 (C++ / LibTorch) 编译与运行说明指南

本文档介绍如何使用 CMake 和 LibTorch 编译并运行从 `gpt.py` 转换而来的 C++ 版本 GPT-2 模型代码（`gpt.cpp`）。

---

## 1. 环境依赖与准备

| 组件 | 最低要求 / 推荐版本 | 说明 |
| :--- | :--- | :--- |
| **操作系统** | Windows 10 / 11 (x64) | 64 位操作系统 |
| **C++ 编译器** | 支持 C++20 的 MSVC（推荐 Visual Studio 2022） | 本项目使用 C++20；PyTorch 各版本的要求见[官方兼容矩阵](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)。 |
| **构建工具** | CMake >= 3.18 | 需配置在系统 `PATH` 环境变量中 |
| **核心依赖** | LibTorch (CPU 或 CUDA) | 可使用独立安装包，也可使用 Python PyTorch 包中附带的 C++ 库，见第 1.3 节 |

> [!NOTE]
> 新构建目录默认使用当前 `PATH` 中的 Python，并从该解释器的 PyTorch 包查找 C++ 库。可通过 `-DPython3_EXECUTABLE` 指定解释器，或通过 `-DCMAKE_PREFIX_PATH`、`-DTorch_DIR` 手动选择独立 LibTorch。原有构建目录的 CMake 缓存会保留之前的选择；切换 Python 或库时请使用新构建目录。

### 1.1 下载 LibTorch

以下以 Windows x64、CPU、Release 版 LibTorch 2.14.1 为例，下载地址来自 PyTorch 官方：

- [LibTorch 2.14.1+cpu（Windows x64，包含依赖的动态库）](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.14.1%2Bcpu.zip)
- [LibTorch CPU 版本下载目录](https://download.pytorch.org/libtorch/cpu/)，可查找其他版本和平台。
- [PyTorch C++ 安装文档](https://docs.pytorch.org/cppdocs/installing.html)，介绍 LibTorch 和 CMake 的基本用法。

上述链接和政策于 2026-10-28 核对。`+cpu` 表示该包使用 CPU，不包含 CUDA 加速支持。

建议将压缩包解压到独立目录，保留其他项目可能正在使用的旧版本。例如，解压后使用以下目录：

```text
F:\LLM\libtorch-2.14.1-cpu\libtorch
```

`CMAKE_PREFIX_PATH` 应指向包含 `include`、`lib` 和 `share` 的内层 `libtorch` 目录。下载路径是单独的 LibTorch 安装；它不会自动替换 Python 的 `torch` 包。

可以分别检查 Python 和 LibTorch 的版本，以及当前 CMake 构建目录缓存的 Torch 路径：

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
Get-Content 'F:\LLM\libtorch-2.14.1-cpu\libtorch\build-version'
Select-String -Path '.\build-2.14.1\CMakeCache.txt' -Pattern '^Torch_DIR:PATH='
```

最后一条命令需在下文的 CMake 配置成功后、从 `gpt2cpp` 目录执行。此机器在 2026-10-28 检查时，Python PyTorch 是 `2.14.1+cpu`，原有 `F:\LLM\libtorch` 是 `2.10.0+cpu`；实际环境可能已变化，请以检查结果为准。

对齐版本有助于排查差异，但相同随机种子和版本并不保证模型参数或输出相同：还需要检查模型权重、随机初始化顺序、设备和推理设置。本项目 [`gpt.cpp`](gpt.cpp) 的前馈层显式按顺序构造各层，以避免 C++ 函数参数求值顺序影响初始化。

### 1.2 Windows Debug 预编译包的发布政策

PyTorch 从 **2.12.1** 起停止发布 Windows LibTorch 的预编译 Debug 压缩包；Windows Release 包、Python wheel 及其他平台的 LibTorch 包不受此政策影响。[PyTorch 官方公告](https://dev-discuss.pytorch.org/t/important-note-stop-publishing-prebuilt-windows-libtorch-debug-binaries/3400)

官方列出的原因包括 Debug 包下载量低，以及 Debug CRT 和第三方依赖带来的维护成本较高。

使用上面的 Release 包时，请选择与其运行时兼容的配置，例如 MSVC 的 `Release` 或使用 `/MD` 的 `RelWithDebInfo`。应用程序的 `RelWithDebInfo` 能为应用自身生成调试信息，但不保证预编译 LibTorch 附带完整调试符号。若确实需要使用 Debug CRT 的 LibTorch，官方建议从源码构建；参见[官方公告中的源码构建说明](https://dev-discuss.pytorch.org/t/important-note-stop-publishing-prebuilt-windows-libtorch-debug-binaries/3400)和 [PyTorch 源码构建说明](https://github.com/pytorch/pytorch#from-source)。

### 1.3 使用 Python PyTorch 包中的 C++ 库

已通过 pip 安装的 PyTorch 也可以提供 C++ 头文件、库文件和 CMake 配置；不一定需要另行下载独立的 LibTorch 压缩包。[PyTorch C++ 安装文档](https://docs.pytorch.org/cppdocs/installing.html)说明可以用 `torch.utils.cmake_prefix_path` 找到 pip 包中的 CMake 配置。

本机检查时，Python 使用 `2.14.1+cpu`，其 `torch` 包位于：

```text
C:\Users\qiuxueliang\AppData\Roaming\Python\Python313\site-packages\torch
├── include\                         C++ 头文件
├── lib\                             torch、torch_cpu、c10 等 .lib 和 .dll
└── share\cmake\Torch\TorchConfig.cmake
```

可用下面的命令查看当前 `python` 实际使用的解释器、PyTorch 版本和 CMake 路径；不同 Python 环境的结果可能不同：

```powershell
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CMake:', torch.utils.cmake_prefix_path)"
```

第 2 节方法 C 使用 CMake 默认自动发现该路径。运行时需要找到同一套 PyTorch 版本的 DLL；本项目的 CMake 配置会把选中包的 `lib` 目录复制到程序输出目录。构建和运行时应避免混用不同版本的 LibTorch 头文件、链接库和 DLL。

本项目的 [`TiktokenDecoder.cpp`](TiktokenDecoder.cpp) 还会嵌入 Python 并导入 `tiktoken`，因此构建时需要 Python 开发文件，运行时也需要相应的 Python 环境和 `tiktoken`。CMake 发现的解释器用于配置和链接；嵌入式 Python 的运行时模块搜索路径可能不同，使用虚拟环境时需另行验证 `tiktoken` 可被实际运行的程序导入。

**本机验证记录（2026-10-28）：** Windows x64、Visual Studio 2022、Release 配置，通过本项目的 `CMakeLists.txt` 在独立临时构建目录中使用 Python PyTorch `2.14.1+cpu` 包中的库，完成配置、编译、链接和运行。`Torch_DIR` 指向该 Python 包的 CMake 目录；输出目录中的 `torch_cpu.dll` 与包内文件的 SHA-256 一致。输入 `Hello, I am` 时，C++ 和 Python 的 token 输出均为：

```text
15496  11  314  716  27018  24086  47843  30961  42348  7267
```

配置时出现 `library kineto not found` 警告；该 Python 包中未发现 `kineto.lib`，但本次 GPT-2 CPU 示例的编译、链接和运行均成功。其他功能是否受影响需要针对具体用途验证。以上一致性只覆盖这次测试，不保证其他输入或环境中的所有浮点结果逐位一致。

---

## 2. 快速开始 (推荐步骤)

### 方法 A：一键双击运行批处理脚本（推荐）
在 `gpt2cpp` 目录下，直接双击运行：
👉 **[`generate_sln.bat`](generate_sln.bat)**

无参数运行时，脚本在 `build-python-torch` 中生成解决方案，默认使用当前 Python 环境的 PyTorch。要选择独立 LibTorch，可在终端中将其目录作为第一个参数：

```powershell
.\generate_sln.bat "F:\LLM\libtorch"
```

手动模式默认使用 `build-libtorch`。还可以通过第二个参数指定其他构建目录；切换 Python 或 LibTorch 路径时，请选择新的构建目录，避免混用之前缓存的库。

---

### 方法 B：命令行使用独立 LibTorch
打开 PowerShell 或 Windows Terminal：

#### 步骤 1：进入源码目录
```powershell
cd F:\LLM\BuildLLM\LLMs-from-scratch\gpt2cpp
# 如果你的工程在 H 盘，请切换为：
# cd H:\LLM\BuildLLM\LLMs-from-scratch\gpt2cpp
```

#### 步骤 2：生成 CMake 工程
使用 Visual Studio 2022 生成 x64 项目：
```powershell
# 假设 2.14.1 已按第 1.1 节解压；也可以替换成实际的 LibTorch 路径。
cmake -S . -B build-2.14.1 -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_PREFIX_PATH=F:/LLM/libtorch-2.14.1-cpu/libtorch" -DUSE_CUDA=OFF
```

使用新的构建目录可以避免旧版 LibTorch 的 `Torch_DIR` 缓存影响查找。配置成功后，可按第 1.1 节的命令核对缓存中的实际路径。

#### 步骤 3：编译 Release 版本
```powershell
cmake --build build-2.14.1 --config Release
```
> [!TIP]
> `CMakeLists.txt` 中配置了 `POST_BUILD` 自动复制命令，编译完成后会自动把 LibTorch 的动态链接库（`.dll`）复制到可执行程序目录，无需手动配置系统环境变量。

#### 步骤 4：运行程序
```powershell
.\build-2.14.1\Release\gpt.exe
```

---

### 方法 C：使用 Python PyTorch 包中的 C++ 库

在已安装 PyTorch 和 `tiktoken` 的 Python 环境中，从本项目的 `gpt2cpp` 目录运行：

```powershell
cmake -S . -B build-python-torch -G "Visual Studio 17 2022" -A x64 -DUSE_CUDA=OFF
cmake --build build-python-torch --config Release
.\build-python-torch\Release\gpt.exe
```

CMake 默认查找 `PATH` 中的 `python`，然后通过该解释器的 `torch.utils.cmake_prefix_path` 找到 PyTorch。如需指定其他解释器，可在**新构建目录**的配置命令中加入 `-DPython3_EXECUTABLE="C:/path/to/python.exe"`。如果该环境没有安装 `torch`，配置会报错并提示可选处理方式，不会自动回退到旧版独立 LibTorch。

选择新的构建目录可避免旧的 `Torch_DIR`、Caffe2 和链接库缓存。配置成功后，可用以下命令检查实际使用的是哪个 Torch 配置：

```powershell
Select-String -Path '.\build-python-torch\CMakeCache.txt' -Pattern '^Torch_DIR:PATH='
```

---

## 3. 运行输出示例

以下是使用 LibTorch 2.10.0 和旧版前馈层初始化写法的历史输出示例。当前源码、权重初始化顺序或 PyTorch/LibTorch 版本变化后，生成 token 可能不同；实际结果以本地运行为准。

```text
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: [1, 4]


==================================================
                      OUT
==================================================

Output: 
 15496     11    314    716  23307  45770  45653   1949  38428  46646
[ CPULongType{1,10} ]
Output length: 10
Output text: Hello, I amToo nexus obe try Pepsi KM
```

---

## 4. 常见问题排查 (FAQ)

### Q1: 运行报找不到 `torch_cpu.dll` 或 `c10.dll`？
- **原因**：动态库未与可执行文件同目录且未加入系统 `PATH`。
- **解决办法**：
  检查实际构建目录的 `Release/` 下是否存在 `torch_cpu.dll`、`c10.dll` 等文件。如果缺失，以下按第 2 节的路径手动复制：
  ```powershell
  Copy-Item -Path "F:\LLM\libtorch-2.14.1-cpu\libtorch\lib\*.dll" -Destination ".\build-2.14.1\Release\"
  ```

### Q2: 提示 Debug 与 Release 运行时不匹配？
- **原因**：Windows 下 LibTorch 的 Release 构建使用 Release CRT；默认使用 Debug CRT 的构建与其不兼容（`/MD` 与 `/MDd` 不匹配）。
- **解决办法**：
  使用 `--config Release`，或使用运行时设置为 `/MD` 的 `RelWithDebInfo`。如需 Debug CRT，请参考第 1.2 节自行构建 Debug 版 LibTorch。

### Q3: 提示 `fatal error C1128: 节数超过对象文件格式限制: 请使用 /bigobj 进行编译`？
- **原因**：PyTorch C++ 模板展开非常庞大，超过了 COFF 默认的 65,536 个节限制。
- **解决办法**：
  本项目 `CMakeLists.txt` 中已默认配置 `target_compile_options(gpt PRIVATE /bigobj)`，如使用其他自定义构建脚本请确保开启 `/bigobj`。
