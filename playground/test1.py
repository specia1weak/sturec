import torch
import torch.nn as nn

"""
均值分支 + 方差分支
均值分支为当前batch的所有样本输出同一个值(只要不训练, 他甚至不会动)
方差分支在batch层面的均值总是0, 这保证了他不背均值打扰
"""
class BifurcatedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, zero_mean_dim='batch'):
        """
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param zero_mean_dim: 'feature' (对每个样本的特征向量求0均值)
                           或 'batch' (对Batch内的特定特征位求0均值)
        """
        super().__init__()
        self.output_dim = output_dim
        self.zero_mean_dim = zero_mean_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # 定义 0 均值化的方法 (关闭 affine 避免模型偷偷把非零均值学回来)
        if self.zero_mean_dim == 'feature':
            self.zero_mean_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        elif self.zero_mean_dim == 'batch':
            self.zero_mean_norm = nn.BatchNorm1d(output_dim, affine=False)
        else:
            raise ValueError("zero_mean_dim must be 'feature' or 'batch'")

        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        raw_fluctuation = self.linear(x)
        fluctuation = self.zero_mean_norm(raw_fluctuation)  # shape: [Batch, OutputDim]
        bias_expanded = self.bias.unsqueeze(0).expand(batch_size, -1)
        output = torch.stack([bias_expanded, fluctuation], dim=1)
        return output


# ================= 实验测试 =================
if __name__ == "__main__":
    batch_size = 4
    input_dim = 16
    output_dim = 8

    # 模拟输入数据
    x = torch.randn(batch_size, input_dim)

    # 实例化我们的分叉网络 (以 Feature 维度 0 均值为例)
    expert_layer = BifurcatedLinear(input_dim, output_dim, zero_mean_dim='batch')

    # 前向传播
    out = expert_layer(x)

    print(f"Output Shape: {out.shape}")  # 预期: [4, 2, 8]

    # 拆解验证
    bias_part = out[:, 0, :]  # 取出 Bias 部分
    fluctuation_part = out[:, 1, :]  # 取出 Fluctuation 部分

    print("\n--- 验证物理意义 ---")
    print(f"Bias 部分是否在 Batch 维度上完全一致? : {torch.allclose(bias_part[0], bias_part[1])}")

    # 验证 fluctuation 在 feature 维度上的均值是否接近 0 (精度误差内)
    mean_of_fluctuation = fluctuation_part.mean(dim=-1)
    print(f"Fluctuation 部分的均值 (应极度接近 0): \n{mean_of_fluctuation.detach().numpy()}")