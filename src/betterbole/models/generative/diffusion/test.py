from betterbole.models.generative.diffusion.base import PredX0Model, PredVModel
from betterbole.models.generative.diffusion.schedulers import DDIMScheduler
from betterbole.models.generative.diffusion.diffusions import CDCDRMlpDiffusion
import torch.nn as nn
import torch.nn.functional as F
import torch

import math


def test_cdcdr_diffusion_refactored():
    print(">>> 初始化测试环境...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数配置
    batch_size = 4
    embedding_size = 64
    num_train_timesteps = 200

    # 模拟输入数据
    # pos_item_e: 目标物品特征 (x_0)
    # UI_aggregation_e: 聚合后的用户特征 (条件 y)
    pos_item_e = torch.randn(batch_size, embedding_size, device=device)
    UI_aggregation_e = torch.randn(batch_size, embedding_size, device=device)

    # 初始化训练调度器和模型
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule_type="linear",
        beta_start=0.0001,
        beta_end=0.02
    )
    model = CDCDRMlpDiffusion(scheduler, embedding_size, uncon_p=0.1).to(device)
    print("\n>>> 开始测试 [训练阶段]...")
    loss = model.calculate_loss(x_0=pos_item_e, y=UI_aggregation_e)
    print(f"Training MSE Loss: {loss.item():.4f}")

    # ==========================================
    # 2. 推理阶段测试：DDPM (全步数去噪)
    # ==========================================
    print("\n>>> 开始测试 [DDPM 推理阶段]...")
    model.eval()
    # 初始化纯噪声 x_T
    x_T = torch.randn_like(pos_item_e)
    # 获取 CFG 所需的空条件
    null_y = model.get_null_condition(batch_size, device)

    # 运行父类的去噪引擎
    predicted_item_ddpm, _ = model.denoise(
        x=x_T,
        y=UI_aggregation_e,
        null_y=null_y,
        guidance_scale=3.0,
        num_inference_steps=200
    )
    print(f"DDPM 去噪完成，输出 shape: {predicted_item_ddpm.shape}")

    # ==========================================
    # 3. 推理阶段测试：DDIM (跳步加速去噪)
    # ==========================================
    print("\n>>> 开始测试 [DDIM 推理阶段]...")

    # DDIM 跳步加速：只需 20 步
    predicted_item_ddim, _ = model.denoise(
        x=x_T,
        y=UI_aggregation_e,
        null_y=null_y,
        guidance_scale=3.0,
        num_inference_steps=3
    )

    print(f"DDIM 去噪完成，输出 shape: {predicted_item_ddim.shape}")
    print(f"DDIM 实际调度的时间步序列: {scheduler.timesteps.tolist()}")


if __name__ == "__main__":
    test_cdcdr_diffusion_refactored()