# Docker 环境配置指南

如果你希望隔离项目的依赖和配置，Docker 是一种有效的开发环境方案。它无需手动安装软件包和库，并能确保开发环境一致。

如果你更偏好 Docker，而非 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 和 [../02_installing-python-libraries](../02_installing-python-libraries) 中介绍的 conda 方法，本指南会带你为本书配置可选的 Docker 环境。

<br>

## 下载并安装 Docker

开始使用 Docker 最简单的方法，是安装适合你所在平台的 [Docker Desktop](https://docs.docker.com/desktop/)。

Linux（Ubuntu）用户也可以选择安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)，并完成[安装后的配置步骤](https://docs.docker.com/engine/install/linux-postinstall/)。

<br>

## 在 Visual Studio Code 中使用 Docker DevContainer

Docker DevContainer（开发容器）允许开发者将 Docker 容器用作完整的开发环境。无论本地计算机如何配置，这种方式都能让用户快速获得一致的开发环境。

DevContainer 也适用于其他 IDE，不过常用的 IDE／编辑器是 Visual Studio Code（VS Code）。下面以 VS Code 为例，介绍如何使用本书的 DevContainer；PyCharm 的流程应当类似。如果尚未安装且希望使用 VS Code，请先[安装](https://code.visualstudio.com/download)。

1. 克隆本 GitHub 仓库，并用 `cd` 进入项目根目录。

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. 将 `setup/03_optional-docker-environment/` 中的 `.devcontainer` 目录移到当前目录（项目根目录）。

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. 在 Docker Desktop 中，确保 **_desktop-linux_ 构建器**正在运行，并用于构建 Docker 容器（依次进入 _Docker Desktop_ → _Change settings_ → _Builders_ → _desktop-linux_ → _..._ → _Use_）。

4. 如果你有[支持 CUDA 的 GPU](https://developer.nvidia.com/cuda-gpus)，可以加速训练和推理：

    4.1 按[此处](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)说明安装 **NVIDIA Container Toolkit**。其支持情况见[此处](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2)。

    4.2 在 Docker Engine 守护进程配置中添加 _nvidia_ 运行时（依次进入 _Docker Desktop_ → _Change settings_ → _Docker Engine_）。将以下内容添加到配置中：

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    例如，完整的 Docker Engine 守护进程 JSON 配置应类似于：

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    然后重启 Docker Desktop。

5. 在终端输入 `code .`，用 VS Code 打开项目。也可以启动 VS Code，在界面中选择要打开的项目。

6. 在 VS Code 左侧的 _Extensions（扩展）_ 菜单中，安装 **Remote Development** 扩展。

7. 打开 DevContainer。

由于 `LLMs-from-scratch` 主目录中有 `.devcontainer` 文件夹（根据系统设置，以 `.` 开头的文件夹可能不可见），VS Code 应会自动检测到它，并询问是否在开发容器中打开项目。如果没有提示，请按 `Ctrl + Shift + P` 打开命令面板，输入 `dev containers`，查看所有 DevContainer 专用选项。


&nbsp;
> ⚠️ **关于以 root 身份运行的说明**
>
> 默认情况下，DevContainer 以 *root 用户*运行。出于安全考虑，通常不建议这样做；不过，为简化本书的环境配置，这里使用 root 配置，以确保全部必需包都能顺利安装到容器中。
>
> 如果你在容器内手动启动 Jupyter Lab，可能会看到以下错误：
>
>   ```bash
>   Running as root is not recommended. Use --allow-root to bypass.
>   ```
>
>   此时，可以运行：
>
>   ```bash
>   uv run jupyter lab --allow-root
>   ```
>
> - 在 VS Code 中使用 Jupyter 扩展时，通常无需手动启动 Jupyter Lab。通过扩展打开 Notebook 应当可以直接使用。
> - 如果高级用户希望采用更严格的安全配置，可以修改 `.devcontainer.json`，设置非 root 用户；但这需要额外配置，大多数使用场景无需如此。



8. 选择 **Reopen in Container（在容器中重新打开）**。

如果此前尚未构建，Docker 会开始构建 `.devcontainer` 配置指定的镜像；如果镜像已在镜像仓库中提供，则会拉取它。

整个过程会自动进行，根据系统和网络速度，可能需要几分钟。可以点击 VS Code 右下角的“Starting Dev Container (show log)”，查看当前构建进度。

完成后，VS Code 会自动连接到容器，并在新建的 Docker 开发环境中重新打开项目。你可以像在本地计算机上运行一样编写、执行和调试代码，同时享受 Docker 提供的隔离性与一致性。

&nbsp;
> **警告：**
> 如果在构建过程中遇到错误，很可能是计算机没有兼容的 GPU，因此不支持 NVIDIA Container Toolkit。此时，请编辑 `devcontainer.json`，删除 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 这一行，再次执行“Reopen Dev Container”流程。

9. 完成。

镜像拉取并构建完成后，项目应已挂载到容器中，所有包也已安装，可以开始开发。

<br>

## 移除 Docker 镜像

如果不再使用，以下步骤介绍如何移除 Docker 容器和镜像。此过程不会从系统卸载 Docker，只会清理本项目相关的 Docker 产物。

1. 列出全部 Docker 镜像，找到与 DevContainer 对应的镜像：

```bash
docker image ls
```

2. 使用镜像 ID 或名称移除 Docker 镜像：

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## 卸载 Docker

如果决定不再使用 Docker 并希望卸载，请参阅[官方文档](https://docs.docker.com/desktop/uninstall/)，其中介绍了各操作系统的具体步骤。
