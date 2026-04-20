from abc import abstractmethod
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLossType(Enum):
    VANILLA = "vanilla"
    LSGAN = "lsgan"
    WGAN = "wgan"
    HINGE = "hinge"


class BaseGAN(nn.Module):
    loss_type = GANLossType.VANILLA

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        if not isinstance(self.generator, nn.Module) or not isinstance(self.discriminator, nn.Module):
            raise TypeError("build_generator 和 build_discriminator 必须返回 nn.Module 实例。")

    @abstractmethod
    def build_generator(self) -> nn.Module:
        """必须实现：返回生成器网络"""
        pass

    @abstractmethod
    def build_discriminator(self) -> nn.Module:
        """必须实现：返回判别器网络"""
        pass

    def forward(self, z, y=None):
        """
        推理专用：给定潜变量生成数据。
        """
        return self.generator(z, y) if y is not None else self.generator(z)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, y=None):
        was_training = self.training
        self.eval()  # 切换到推理模式
        try:
            z = torch.randn(batch_size, self.latent_dim, device=device)
            return self.forward(z, y)
        finally:
            # 2. 无论是否发生异常，强制恢复之前的状态
            if was_training:
                self.train()

    # ==========================================
    # 以下为被锁死的固定逻辑：损失计算与交替训练
    # ==========================================

    def compute_d_loss(self, real_data, fake_data, y=None):
        """计算判别器损失"""
        real_pred = self.discriminator(real_data, y) if y is not None else self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach(), y) if y is not None else self.discriminator(fake_data.detach())

        if self.loss_type == GANLossType.VANILLA:
            d_loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            return (d_loss_real + d_loss_fake) / 2

        elif self.loss_type == GANLossType.LSGAN:
            d_loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
            d_loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            return (d_loss_real + d_loss_fake) / 2

        elif self.loss_type == GANLossType.HINGE:
            d_loss_real = torch.mean(F.relu(1.0 - real_pred))
            d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
            return d_loss_real + d_loss_fake

        elif self.loss_type == GANLossType.WGAN:
            return -(torch.mean(real_pred) - torch.mean(fake_pred))
        else:
            raise NotImplementedError(f"Unsupported loss type: {self.loss_type}")

    def compute_g_loss(self, fake_data, y=None):
        """计算生成器损失"""
        fake_pred = self.discriminator(fake_data, y) if y is not None else self.discriminator(fake_data)

        if self.loss_type == GANLossType.VANILLA:
            return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

        elif self.loss_type == GANLossType.LSGAN:
            return F.mse_loss(fake_pred, torch.ones_like(fake_pred))

        elif self.loss_type == GANLossType.HINGE:
            return -torch.mean(fake_pred)

        elif self.loss_type == GANLossType.WGAN:
            return -torch.mean(fake_pred)
        else:
            raise NotImplementedError()

    def optimize_step(self, real_data, opt_G, opt_D, y=None):
        """
        万能训练单步逻辑。锁定 detach() 的易错点和计算流程。
        外部训练循环只需调用此方法。
        """
        batch_size = real_data.size(0)
        device = real_data.device

        # ==========================================
        # 1. 训练判别器 (Discriminator)
        # ==========================================
        opt_D.zero_grad()
        z_d = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data_d = self.generator(z_d, y).detach() if y is not None else self.generator(z_d).detach()
        d_loss = self.compute_d_loss(real_data, fake_data_d, y)
        d_loss.backward()
        opt_D.step()

        # ==========================================
        # 2. 训练生成器 (Generator)
        # ==========================================
        opt_G.zero_grad()
        z_g = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data_g = self.generator(z_g, y) if y is not None else self.generator(z_g)
        g_loss = self.compute_g_loss(fake_data_g, y)
        g_loss.backward()
        opt_G.step()

        return d_loss.item(), g_loss.item()