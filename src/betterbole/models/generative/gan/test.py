import math
import torch
from torch import nn
from betterbole.models.generative.gan.gans import VectorResGAN
import matplotlib.pyplot as plt

from torch.nn.utils import spectral_norm

# ==========================================
# 2. 模拟真实数据生成器
# ==========================================
def get_real_data_batch(batch_size, data_dim, device):
    """
    生成具有规律的 64 维向量：带有随机相位的正弦波，并加入微小噪声。
    这样可以测试模型是否真的学到了数据分布（而不是单纯记住了几个点）。
    """
    # 生成时间步 t (长度为 data_dim)
    t = torch.linspace(0, 4 * math.pi, data_dim).unsqueeze(0).repeat(batch_size, 1).to(device)
    # 为每个 batch 生成随机相位
    phases = (torch.rand(batch_size, 1) * 2 * math.pi).to(device)
    # 计算正弦波并加入少量高斯噪声
    clean_wave = torch.sin(t + phases)
    noise = torch.randn_like(clean_wave) * 0.05
    # Tanh 激活函数的输出域是 [-1, 1]，确保真实数据也在此范围
    real_data = torch.clamp(clean_wave + noise, -1.0, 1.0)
    return real_data


# ==========================================
# 3. Main 训练脚本
# ==========================================
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LATENT_DIM = 32
    DATA_DIM = 64
    BATCH_SIZE = 128
    EPOCHS = 2000

    # 0. 计算向量标准化信息
    from betterbole.models.generative.scaler.scalers import StreamingStandardScaler
    scaler = StreamingStandardScaler().to(device)
    for _ in range(1000):
        batch = get_real_data_batch(128, DATA_DIM, device)
        scaler.collect(batch)
    scaler.finalize()

    # 1. 实例化模型
    model = VectorResGAN(latent_dim=LATENT_DIM, data_dim=DATA_DIM).to(device)
    opt_G = torch.optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(model.discriminator.parameters(), lr=4e-4, betas=(0.0, 0.9))

    print("Starting Training...")
    model.train()

    # 3. 训练循环
    for epoch in range(EPOCHS):
        real_data = scaler.forward(get_real_data_batch(BATCH_SIZE, DATA_DIM, device))
        d_loss, g_loss = model.optimize_step(real_data, opt_G, opt_D)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1:4d}/{EPOCHS}] | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

    print("Training finished. Plotting results...")

    # ==========================================
    # 4. 可视化测试
    # ==========================================
    # 采样 3 条假数据曲线和 3 条真数据曲线做对比
    model.eval()
    with torch.no_grad():
        fake_samples = model.sample(batch_size=3, device=device)
        fake_samples = scaler.inverse(fake_samples).cpu().numpy()
        real_samples = get_real_data_batch(3, DATA_DIM, device).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(3):
        axs[0].plot(real_samples[i], label=f'Real Sample {i + 1}')
    axs[0].set_title("Real Data (Sine waves w/ random phase)")
    axs[0].set_ylim(-1.2, 1.2)

    for i in range(3):
        axs[1].plot(fake_samples[i], label=f'Generated Sample {i + 1}')
    axs[1].set_title("Generated Data (from random noise)")
    axs[1].set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()