from torch import nn

from betterbole.models.backbone.common import build_expert_factory, build_tower_factory, to_dims
from betterbole.models.utils.container import MultiScenarioContainer as MSC

class SharedBottomLess(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        self.expert = build_expert_factory(emb_size, expert_dims)()
        self.towers = MSC(num_domains, build_tower_factory(expert_dims[-1]))

    @staticmethod
    def default_expert_dims(emb_size):
        return (max(1, int(emb_size)),)

    def forward(self, x, domain_ids):
        expert_out = self.expert(x)
        return self.towers.forward(expert_out, domain_ids).squeeze(-1)

class SharedBottomPlus(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        self.expert = build_expert_factory(emb_size, expert_dims)()
        self.towers = MSC(num_domains, build_tower_factory(expert_dims[-1]))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (emb_size, emb_size, max(1, emb_size // 2), emb_size)

    def forward(self, x, domain_ids):
        expert_out = self.expert(x)
        return self.towers.forward(expert_out, domain_ids).squeeze(-1)

if __name__ == '__main__':
    from betterbole.models.utils.tests import dummy_input_multi_domain
    x, dids = dummy_input_multi_domain(6)
    model = SharedBottomLess(128, 6,)
    ret = model.forward(x, dids)
    print(ret.shape)





