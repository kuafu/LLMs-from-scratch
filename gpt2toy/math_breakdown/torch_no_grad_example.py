"""演示 no_grad 的梯度记录、状态恢复与模型推理。"""

import torch
from torch import nn


def main():
    x = torch.tensor([2.0], requires_grad=True)
    print("进入前:", torch.is_grad_enabled())
    print(x.shape)
    print(x.size())

    with torch.no_grad():
        y = x * 3
        print("代码块内:", torch.is_grad_enabled())
        print("y:", y.item(), "requires_grad:", y.requires_grad)

    print("退出后:", torch.is_grad_enabled())
    z = x * 4
    z.backward()
    print("z:", z.item(), "x.grad:", x.grad.item())

    # eval() 控制层的行为；no_grad() 控制是否记录梯度。
    model = nn.Linear(2, 1)
    model.eval()
    input_data = torch.tensor([[1.0, 2.0]])  # shape: (1, 2)
    with torch.no_grad():
        prediction = model(input_data)
    print("预测结果形状:", tuple(prediction.shape))  # (1, 1)
    print("预测结果记录梯度:", prediction.requires_grad)  # False


if __name__ == "__main__":
    main()
