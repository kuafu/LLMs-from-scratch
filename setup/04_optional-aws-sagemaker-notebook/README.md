# AWS CloudFormation 模板：预装 LLMs-from-scratch 仓库的 Jupyter Notebook

此 CloudFormation 模板会在 Amazon SageMaker 中创建支持 GPU 的 Jupyter Notebook 实例，并配置执行角色和 LLMs-from-scratch GitHub 仓库。

## 模板的功能：

1. 创建 IAM 角色，授予 SageMaker Notebook 实例所需权限。
2. 创建 KMS 密钥及其别名，用于加密 Notebook 实例。
3. 配置 Notebook 实例的生命周期配置脚本，执行以下操作：
   - 在用户主目录中单独安装 Miniconda。
   - 创建自定义 Python 环境，安装支持 CUDA 的 TensorFlow 2.15.0 和 PyTorch 2.1.0。
   - 安装 Jupyter Lab、Matplotlib 及其他实用库。
   - 将自定义环境注册为 Jupyter 内核。
4. 按指定配置创建 SageMaker Notebook 实例，包括支持 GPU 的实例类型、执行角色和默认代码仓库。

## 使用方法：

1. 下载 CloudFormation 模板文件（`cloudformation-template.yml`）。
2. 在 AWS 管理控制台中，进入 CloudFormation 服务。
3. 创建新的堆栈并上传模板文件。
4. 提供 Notebook 实例名称（例如“LLMsFromScratchNotebook”）（默认代码仓库为 LLMs-from-scratch GitHub 仓库）。
5. 检查并接受模板参数，然后创建堆栈。
6. 堆栈创建完成后，可在 SageMaker 控制台中找到该 Notebook 实例。
7. 打开 Notebook 实例，在预配置的环境中开展 LLMs-from-scratch 项目。

## 要点：

- 模板会创建支持 GPU 的 Notebook 实例（`ml.g4dn.xlarge`），存储空间为 50GB。
- 它会配置自定义 Miniconda 环境，安装支持 CUDA 的 TensorFlow 2.15.0 和 PyTorch 2.1.0。
- 自定义环境会注册为 Jupyter 内核，供 Notebook 使用。
- 模板还会创建用于加密 Notebook 实例的 KMS 密钥，以及具有所需权限的 IAM 角色。
