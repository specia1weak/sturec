from typing import List, Iterable
import torch
from torch import nn

from betterbole.models.utils.activation import activation_layer

class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', dropout_rate: float=0., use_bn=False, dice_dim=2):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

# class MLP(nn.Module):
#     def __init__(self, *dims, dropout_rate: float = 0.0, activation:str = 'relu', batch_norm=False):
#         super().__init__()
#         self.net = DNN(dims[0], dims[1:], activation, dropout_rate=dropout_rate, use_bn=batch_norm)
#
#     def forward(self, x):
#         return self.net(x)


class MLP(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation: str = 'relu', batch_norm: bool = False):
        """
        Args:
            *dims: 动态传入各个层的维度。例如 MLP(256, 128, 64) 代表 256->128->64 的两层网络。
            dropout_rate: 丢弃率
            activation: 激活函数名称 ('relu', 'prelu', 'sigmoid', 'linear')
            batch_norm: 是否在隐层使用 BatchNorm1d
        """
        super().__init__()

        if len(dims) < 2:
            raise ValueError("MLP 至少需要输入维度和输出维度，例如 MLP(256, 128)")

        layers = []
        # 2. 遍历构建网络层
        for i in range(len(dims) - 1):
            is_last_layer = (i == len(dims) - 2)

            # (1) 线性映射
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # (2) 隐层处理（最后一层通常不需要加 BN、激活和 Dropout，直接输出 Logits）
            if not is_last_layer:
                # 标准范式：Linear -> BN -> Activation -> Dropout
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                layers.append(activation_layer(activation))

                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))

        # 使用 nn.Sequential 自动管理前向传播
        self.net = nn.Sequential(*layers)

        # 3. 科学的参数初始化（替换掉 DeepCTR 里那个极慢的 0.0001）
        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation.lower() in ['relu', 'prelu']:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn


class FeatureBifurcator(nn.Module):
    def __init__(self, num_features, feature_dim=-1, mean_dims=0, normalize_var=False, stack_dim=1):
        """
        通用特征分化组件：将任意特征分离为“静态可学习基准(Bias)”和“零均值动态波动(Fluctuation)”

        :param num_features: 特征的通道数（用于初始化可学习的静态 Bias）
        :param feature_dim: 特征所在的维度索引。对于 Linear 是 -1，对于 Conv2d(B,C,H,W) 通常是 1
        :param mean_dims: 在哪些维度上计算均值。
                          - Batch级别求均值: 0
                          - 图像空间求均值 (InstanceNorm风格): (2, 3)
                          - 特征内部求均值 (LayerNorm风格): feature_dim
        :param normalize_var: 是否除以标准差（True 则类似原版的 BN/LN，False 则只做纯粹的 0 均值化）
        :param stack_dim: 最终 stack 拼接的维度。默认在 dim=1 堆叠，生成 [..., 2, ...] 的形状
        """
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.mean_dims = mean_dims
        self.normalize_var = normalize_var
        self.stack_dim = stack_dim
        self.eps = 1e-5

        # 初始化静态基准 (可学习的 Bias)
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 1. 提取动态波动 (Fluctuation): 减去均值
        mean_val = x.mean(dim=self.mean_dims, keepdim=True)
        fluctuation = x - mean_val

        # 可选：是否进行方差归一化缩放
        if self.normalize_var:
            var_val = x.var(dim=self.mean_dims, keepdim=True, unbiased=False)
            fluctuation = fluctuation / torch.sqrt(var_val + self.eps)

        # 2. 形状对齐：将 1D bias 动态广播成与输入 x 完全相同的形状
        # 无论 x 是 [B, D] 还是 [B, C, H, W]，自动在 feature_dim 展开
        bias_shape = [1] * x.dim()
        bias_shape[self.feature_dim] = self.num_features
        bias_expanded = self.bias.view(*bias_shape).expand_as(x)

        return bias_expanded, fluctuation

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
        return bias_expanded, fluctuation

class BifurcatedMLP(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation: str = 'relu', batch_norm: bool = False):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP 至少需要输入维度和输出维度，例如 MLP(256, 128)")
        layers = []

        # 2. 遍历构建不包括最后一层的网络层
        for i in range(len(dims) - 1):
            is_last_layer = (i == len(dims) - 2)
            if not is_last_layer:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation_layer(activation))
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))

        self.hidden_net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dims[-2], dims[-1])
        self.bifurcator = FeatureBifurcator(dims[-1])
        self._init_weights(activation)

    @property
    def bias(self):
        return self.bifurcator.bias

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation.lower() in ['relu', 'prelu']:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.bifurcator(self.last_layer(self.hidden_net(x)))

class ModuleFactory:
    @staticmethod
    def build_expert(in_dim: int, hidout_dims=None, dropout_rate=0.1):
        """
        构建专家网络。
        :param output_dim: 如果有输出要求，默认emb // size 改成 传参
        :param in_dim: 输入特征维度
        :param depth_multiplier: 深度倍数。
                                 1 对应单层 PLE 的深度 [D, D/2]
                                 2 对应两层 PLE 的总深度 [D, D, D/2]
        """
        # 初始化输入维度
        hidout_dims = hidout_dims or in_dim // 2
        if not isinstance(hidout_dims, Iterable):
            hidout_dims = [hidout_dims]
        dims = [in_dim, *hidout_dims]
        return lambda: MLP(*dims, dropout_rate=dropout_rate, activation="relu")

    @staticmethod
    def build_gate(in_dim, num_output=None):
        if num_output is None or num_output == 1:
            num_output = 1
            return lambda: nn.Sequential(MLP(in_dim, in_dim // 2, num_output), nn.Sigmoid())
        else:
            return lambda: nn.Sequential(MLP(in_dim, in_dim // 2, num_output), nn.Softmax(-1))

    @staticmethod
    def build_tower(in_dim):
        return lambda: MLP(in_dim, in_dim // 2, 1, dropout_rate=0.2)


if __name__ == '__main__':
    N = 5000  # 样本量
    input_dim = 16  # 特征维度
    X = torch.randn(N, input_dim)

    # 我们强行设定一个极其刁钻的“全局偏置” (真实世界里可能是该场景的基础 CTR)
    TRUE_BIAS = 4.26

    # 真实用户的个性化偏好 (均值期望为 0，因为 X 是标准正态分布)
    # 假设前几个特征对结果有强影响
    TRUE_FLUCTUATION = X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 0.5

    # 最终的 Label = 全局偏置 + 个性化波动 + 少量高斯噪声
    Y = TRUE_BIAS + TRUE_FLUCTUATION + torch.randn(N) * 0.1
    Y = Y.unsqueeze(1)  # shape: [5000, 1]

    # ==========================================
    # 3. 训练模型，见证奇迹
    # ==========================================
    # 构建一个 MLP，输出维度必须是 1 (预测单分)
    model = BifurcatedMLP(16, 64, 32, 1)
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    print("\n=== 开始训练 ===")
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()

        # 前向传播，输出 shape: [Batch, 2, 1]
        out = model(X)

        # 把 Bias 和 Fluctuation 加起来算总预测值
        y_pred = out.sum(dim=1)  # shape: [Batch, 1]

        loss = criterion(y_pred, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/500, Loss: {loss.item():.4f}")

    # ==========================================
    # 4. 硬核解剖：验证解耦是否成功
    # ==========================================
    print("\n=== 验证解耦效果 ===")
    model.eval()
    with torch.no_grad():
        test_out = model(X)

        # 提取两部分
        learned_bias = test_out[:, 0, :]  # shape: [Batch, 1]
        learned_fluctuation = test_out[:, 1, :]  # shape: [Batch, 1]

        # 1. 验证 Bias 是否精准抓住了全局大盘水位
        # 参数 self.bias 和 BatchNorm 中记录的 running_mean 的总和才是物理上的真实 bias
        static_param_bias = model.bias.item()
        print()

        print(f"[验证 1] 设定的真实全局偏置 : {TRUE_BIAS}")
        print(f"[验证 1] 模型学到的总体 Bias: {model.bias.item():.4f}")

        # 2. 验证 Fluctuation 是否真的是 0 均值，且拟合了个性化波动
        mean_fluc = learned_fluctuation.mean().item()
        print(f"\n[验证 2] 预测调节值的均值 (应极度接近 0): {mean_fluc:.6f}")

        # 3. 验证 Fluctuation 的相关性
        # 如果解耦成功，学到的 fluctuation 应该和真值高度正相关
        fluc_flat = learned_fluctuation.squeeze()
        correlation = torch.corrcoef(torch.stack([fluc_flat, TRUE_FLUCTUATION]))[0, 1].item()
        print(f"[验证 3] 学到的波动与真实波动的皮尔逊相关系数: {correlation:.4f} (越接近 1.0 越好)")
