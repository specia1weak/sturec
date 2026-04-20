from torch import nn
from .base import BaseGAN
from torch.nn.utils import spectral_norm


class ResBlock1D(nn.Module):
    """通用的 1D 残差块"""

    def __init__(self, dim, use_sn=False):
        super().__init__()

        # 判别器用 SN，生成器不用
        def sn_linear(in_f, out_f):
            return spectral_norm(nn.Linear(in_f, out_f)) if use_sn else nn.Linear(in_f, out_f)

        self.block = nn.Sequential(
            sn_linear(dim, dim),
            nn.LeakyReLU(0.2, inplace=True),
            sn_linear(dim, dim),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))  # 残差连接


class VectorResGAN(BaseGAN):
    def __init__(self, latent_dim=32, data_dim=64, hidden_dim=256):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        super().__init__(latent_dim)

    def build_generator(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(True),
            ResBlock1D(self.hidden_dim, use_sn=False),
            ResBlock1D(self.hidden_dim, use_sn=False),
            nn.Linear(self.hidden_dim, self.data_dim),
        )

    def build_discriminator(self) -> nn.Module:
        return nn.Sequential(
            # 判别器输入层也建议用 SN
            spectral_norm(nn.Linear(self.data_dim, self.hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(self.hidden_dim, use_sn=True),
            ResBlock1D(self.hidden_dim, use_sn=True),
            spectral_norm(nn.Linear(self.hidden_dim, 1))
        )