import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置字体支持中文与清晰画质
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 1. 准备数据
x = torch.linspace(-4, 4, 1000)

# 精确 GELU: x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
gelu_exact = nn.functional.gelu(x, approximate="none")

# 近似 GELU (GPT-2等模型采用的 tanh 近似): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
gelu_approx = nn.functional.gelu(x, approximate="tanh")

# 对比激活函数: ReLU 与 SiLU/Swish
relu = torch.relu(x)
silu = nn.functional.silu(x)

# 2. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- 左图：GELU 与 ReLU / SiLU 对比 ---
ax1.plot(x.numpy(), relu.numpy(), label="ReLU", color="#94a3b8", linestyle="--", linewidth=2)
ax1.plot(x.numpy(), silu.numpy(), label="SiLU / Swish", color="#10b981", linestyle=":", linewidth=2)
ax1.plot(x.numpy(), gelu_exact.numpy(), label="GELU (Exact)", color="#2563eb", linewidth=2.5)

ax1.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
ax1.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
ax1.set_title("GELU vs ReLU vs SiLU 激活函数对比", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("输入 x", fontsize=12)
ax1.set_ylabel("激活输出 f(x)", fontsize=12)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-1, 4)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)

# --- 右图：负值区域与平滑特性放大分析 ---
x_zoom = torch.linspace(-3, 1.5, 800)
gelu_zoom = nn.functional.gelu(x_zoom, approximate="none")
gelu_approx_zoom = nn.functional.gelu(x_zoom, approximate="tanh")
relu_zoom = torch.relu(x_zoom)

ax2.plot(x_zoom.numpy(), relu_zoom.numpy(), label="ReLU (在0处不可导)", color="#94a3b8", linestyle="--", linewidth=2)
ax2.plot(x_zoom.numpy(), gelu_zoom.numpy(), label="GELU (Exact: x·Φ(x))", color="#2563eb", linewidth=2.5)
ax2.plot(x_zoom.numpy(), gelu_approx_zoom.numpy(), label="GELU (Tanh 近似，GPT常用)", color="#f59e0b", linestyle="-.", linewidth=2)

# 标记局部极小值点 (~ -0.7517, -0.1700)
min_idx = torch.argmin(gelu_zoom)
min_x = x_zoom[min_idx].item()
min_y = gelu_zoom[min_idx].item()

ax2.plot(min_x, min_y, 'ro', markersize=6)
ax2.annotate(f"极小值点: ({min_x:.2f}, {min_y:.2f})\n保留了部分微弱负梯度",
             xy=(min_x, min_y), xytext=(min_x - 1.4, min_y - 0.35),
             arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#ef4444", alpha=0.8))

ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
ax2.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
ax2.set_title("GELU 局部放大：平滑性与负值谷区", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("输入 x", fontsize=12)
ax2.set_ylabel("激活输出 f(x)", fontsize=12)
ax2.set_xlim(-3, 1.5)
ax2.set_ylim(-0.6, 1.5)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)

plt.tight_layout()

# 保存图片到工作区及 Artifact 目录
out_path_workspace = "gelu_plot.png"
artifact_dir = r"C:\Users\Nox\.gemini\antigravity\brain\c5042ee4-ded7-4af7-8328-504ba6abe7f2"
out_path_artifact = os.path.join(artifact_dir, "gelu_plot.png")

plt.savefig(out_path_workspace, bbox_inches='tight')
plt.savefig(out_path_artifact, bbox_inches='tight')
print(f"图像已成功保存至:\n  1. {os.path.abspath(out_path_workspace)}\n  2. {out_path_artifact}")
