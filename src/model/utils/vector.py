import torch
t = torch.randn(4, 64)
ret = torch.split(t, 2, dim=-1)
print(type(ret))