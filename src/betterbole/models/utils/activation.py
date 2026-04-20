# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    if isinstance(act_name, str):
        act_name = act_name.lower()
        act_builder = {
            'sigmoid': lambda: nn.Sigmoid(),
            'linear':  lambda: Identity(),
            'relu':    lambda: nn.ReLU(inplace=True),
            'prelu':   lambda: nn.PReLU(),
            'dice':    lambda: Dice(hidden_size, dice_dim) if hidden_size else ValueError("Dice requires hidden_size"),
            'silu':    lambda: nn.SiLU()
        }
        if act_name in act_builder:
            return act_builder[act_name]()  # 注意这里的 ()，这才是真正执行实例化的地方
        else:
            raise NotImplementedError(f"Activation {act_name} is not supported.")

    elif issubclass(act_name, nn.Module):
        return act_name()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    pass