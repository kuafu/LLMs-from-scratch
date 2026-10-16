# 附录 A：PyTorch 简介

### 本章主要代码

- [code-part1.ipynb](code-part1.ipynb) 包含书中 A.1 至 A.8 节出现的全部代码
- [code-part2.ipynb](code-part2.ipynb) 包含书中 A.9 节出现的全部 GPU 代码
- [DDP-script.py](DDP-script.py) 包含用于演示多 GPU 用法的脚本（注意，Jupyter Notebook 仅支持单个 GPU，因此这里提供的是脚本而非笔记本）。你可以通过 `python DDP-script.py` 运行它。如果你的机器有两个以上的 GPU，可以运行 `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py`。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章练习题答案

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，可通过 PyTorch 的 `torchrun` 命令运行，而无需我们自行通过 `multiprocessing.spawn` 创建和管理多个进程。`torchrun` 命令的优点是会自动处理分布式初始化（包括多节点协调），从而略微简化配置过程。你可以通过 `torchrun --nproc_per_node=2 DDP-script-torchrun.py` 运行此脚本。
