import torch
from torch import nn

from src.model.utils.general import MLP, ModuleFactory as MF
from src.model.utils.multisc import MultiScenarioContainer as MSC

class SingleLayerMMoE(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dim=64):
        super().__init__()
        num_experts = num_domains + 1

        build_silu_expert = lambda: MLP(emb_size, emb_size // 2, expert_dim, activation="silu", batch_norm=True)
        self.layer1_experts = MSC(num_experts, build_silu_expert)
        self.layer1_gates = MSC(num_domains, MF.build_gate(emb_size, num_experts))
        self.towers = MSC(num_domains, MF.build_tower(emb_size // 2))

    def forward(self, x, domain_ids):
        layer1_expert_output = self.layer1_experts.forward(x) # B N D
        layer1_gates = self.layer1_gates.forward(x, domain_ids) # B N
        weights = layer1_gates.unsqueeze(-1) # B K 1
        interest = torch.sum(weights * layer1_expert_output, dim=1)
        return self.towers.forward(interest, domain_ids).squeeze(-1)


if __name__ == '__main__':
    from src.model.utils.tests import dummy_input_multi_domain
    x, dids = dummy_input_multi_domain(6)
    model = SingleLayerMMoE(128, 6,)
    ret = model.forward(x, dids)
    print(ret.shape)





