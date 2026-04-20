from typing import List, Iterable

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

class MLP(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation:str = 'relu', batch_norm=False):
        super().__init__()
        self.net = DNN(dims[0], dims[1:], activation, dropout_rate=dropout_rate, use_bn=batch_norm)

    def forward(self, x):
        return self.net(x)


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