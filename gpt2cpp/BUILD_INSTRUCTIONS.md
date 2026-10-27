# GPT-2 (C++ / LibTorch) 编译与运行说明指南

本文档介绍如何使用 CMake 和 LibTorch 编译并运行从 `gpt.py` 转换而来的 C++ 版本 GPT-2 模型代码（`gpt.cpp`）。

---

## 1. 环境依赖与准备

| 组件 | 最低要求 / 推荐版本 | 说明 |
| :--- | :--- | :--- |
| **操作系统** | Windows 10 / 11 (x64) | 64 位操作系统 |
| **C++ 编译器** | MSVC 2019 / 2022 (v142 或更高) | 需要支持 C++20 标准（LibTorch 2.4+ 依赖 C++20），推荐 Visual Studio 2022 |
| **构建工具** | CMake >= 3.18 | 需配置在系统 `PATH` 环境变量中 |
| **核心依赖** | LibTorch (CPU 或 CUDA) | 位于 `H:\LLM\libtorch` 或 `F:\LLM\libtorch` |

> [!NOTE]
> `CMakeLists.txt` 已内置自动探测逻辑，会优先检测 `H:/LLM/libtorch`，若不存在则探测 `F:/LLM/libtorch`。你也可以通过 `-DCMAKE_PREFIX_PATH` 手动指定绝对路径。

---

## 2. 快速开始 (推荐步骤)

### 方法 A：一键双击运行批处理脚本（推荐）
在 `gpt2cpp` 目录下，直接双击运行：
👉 **[`generate_sln.bat`](file:///H:/LLM/BuildLLM/LLMs-from-scratch/gpt2cpp/generate_sln.bat)**

脚本会自动探测 LibTorch 路径（支持 `H:\LLM\libtorch` 与 `F:\LLM\libtorch`），并在 `build` 目录下生成 `gpt_cpp.sln` 解决方案。随后你可以直接用 Visual Studio 双击打开该 `.sln` 文件进行编译和调试。

---

### 方法 B：命令行逐步构建
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
# 指定 LibTorch 路径生成构建文件（以 H 盘为例，可替换为 F 盘）：
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="H:\LLM\libtorch"
```

#### 步骤 3：编译 Release 版本
```powershell
cmake --build build --config Release
```
> [!TIP]
> `CMakeLists.txt` 中配置了 `POST_BUILD` 自动复制命令，编译完成后会自动把 LibTorch 的动态链接库（`.dll`）复制到可执行程序目录，无需手动配置系统环境变量。

#### 步骤 4：运行程序
```powershell
.\build\Release\gpt.exe
```

---

## 3. 运行输出示例

执行成功后，终端将输出如下文本生成结果：

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
Columns 1 to 11 15496     11    314    716  23307  45770  45653   1949  38428  46646  12919

Columns 12 to 14 33634  40392  40187
[ CPULongType{1,14} ]
Output length: 14
```

---

## 4. 常见问题排查 (FAQ)

### Q1: 运行报找不到 `torch_cpu.dll` 或 `c10.dll`？
- **原因**：动态库未与可执行文件同目录且未加入系统 `PATH`。
- **解决办法**：
  检查 `build/Release/` 目录下是否存在 `torch_cpu.dll`、`c10.dll` 等文件。如果缺失，可手动复制：
  ```powershell
  Copy-Item -Path "H:\LLM\libtorch\lib\*.dll" -Destination ".\build\Release\"
  ```

### Q2: 提示 Debug 与 Release 运行时不匹配？
- **原因**：LibTorch 的 Release 版本库只能与 MSVC Release 配置编译产物链接（`/MD` 与 `/MDd` 冲突）。
- **解决办法**：
  执行编译时务必带上 `--config Release` 参数。

### Q3: 提示 `fatal error C1128: 节数超过对象文件格式限制: 请使用 /bigobj 进行编译`？
- **原因**：PyTorch C++ 模板展开非常庞大，超过了 COFF 默认的 65,536 个节限制。
- **解决办法**：
  本项目 `CMakeLists.txt` 中已默认配置 `target_compile_options(gpt PRIVATE /bigobj)`，如使用其他自定义构建脚本请确保开启 `/bigobj`。
