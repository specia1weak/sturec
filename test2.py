import torch

# 假设 tensor 的形状是未知的多维，这里模拟一个 4 维 tensor (2, 5, 3, 4)
tensor = torch.randn(2, 5, 3, 4)

# 你的 select_id，必须是一维张量，且数据类型必须是 torch.long
select_id = torch.tensor([0, 2, 4], dtype=torch.long)

# 直接在 dim=1 上进行索引提取
result = torch.index_select(tensor, dim=1, index=select_id)

print(result.shape) # 输出: torch.Size([2, 3, 3, 4])