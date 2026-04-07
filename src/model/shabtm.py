import torch
from torch import nn

from src.model.utils.general import MLP, ModuleFactory as MBF
from src.model.utils.multisc import MultiScenarioContainer

class SharedBottomLess(nn.Module):
    def __init__(self, emb_size, num_domains):
        super().__init__()
        self.expert = MBF.build_expert(emb_size, depth_multiplier=2)()
        self.towers = MultiScenarioContainer(num_domains, MBF.build_tower(emb_size))

    def forward(self, x, domain_ids):
        expert_out = self.expert(x)
        return self.towers.forward(expert_out, domain_ids).squeeze(-1)

class SharedBottomPlus(nn.Module):
    def __init__(self, emb_size, num_domains):
        super().__init__()
        self.expert = MLP(emb_size, emb_size, emb_size, emb_size // 2, emb_size)
        self.towers = MultiScenarioContainer(num_domains, MBF.build_tower(emb_size))

    def forward(self, x, domain_ids):
        expert_out = self.expert(x)
        return self.towers.forward(expert_out, domain_ids).squeeze(-1)

if __name__ == '__main__':
    from src.model.utils.tests import dummy_input_multi_domain
    x, dids = dummy_input_multi_domain(6)
    model = SharedBottomLess(128, 6,)
    ret = model.forward(x, dids)
    print(ret.shape)





