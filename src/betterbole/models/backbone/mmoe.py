import torch
from torch import nn

from betterbole.models.backbone.common import build_expert_factory, build_gate_factory, build_tower_factory, to_dims
from betterbole.models.utils.container import MultiTaskContainer as MTC, MultiScenarioContainer as MSC


class SingleLayerMMoE(nn.Module):
    def __init__(
        self,
        emb_size,
        num_domains,
        expert_dims=None,
    ):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        num_experts = num_domains + 1
        rep_dim = expert_dims[-1]

        self.layer1_experts = MSC(num_experts, build_expert_factory(emb_size, expert_dims, batch_norm=False))
        self.layer1_gates = MSC(num_domains, build_gate_factory(emb_size, num_experts))
        self.towers = MSC(num_domains, build_tower_factory(rep_dim))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (emb_size * 2, emb_size * 3)

    def forward(self, x, domain_ids):
        layer1_expert_output = self.layer1_experts.forward(x) # B N D
        layer1_gates = self.layer1_gates.forward(x, domain_ids) # B N
        weights = layer1_gates.unsqueeze(-1) # B K 1
        interest = torch.sum(weights * layer1_expert_output, dim=1)
        return self.towers.forward(interest, domain_ids).squeeze(-1)

class SingleLayerMTLMMoE(nn.Module):
    def __init__(
        self,
        emb_size,
        task_names,
        expert_dims=None,
    ):
        super().__init__()
        num_tasks = len(task_names)
        self.task_names = task_names
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        num_experts = num_tasks + 1
        rep_dim = expert_dims[-1]

        self.layer1_experts = MSC(num_experts, build_expert_factory(emb_size, expert_dims, batch_norm=False))
        self.layer1_gates = MTC(task_names, build_gate_factory(emb_size, num_experts))
        self.towers = MTC(task_names, build_tower_factory(rep_dim))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (emb_size * 2, emb_size * 3)

    def forward(self, x):
        layer1_expert_output = self.layer1_experts.forward(x)  # B N D
        layer1_gates = self.layer1_gates.forward(x)  # Dict[B N]
        ret = {}
        for task in self.task_names:
            weights = layer1_gates[task].unsqueeze(-1)
            interest = torch.sum(weights * layer1_expert_output, dim=1)
            ret[task] = self.towers[task].forward(interest).squeeze(-1)
        return ret

if __name__ == '__main__':
    from betterbole.models.utils.tests import dummy_input_multi_domain
    x, dids = dummy_input_multi_domain(6)
    model = SingleLayerMTLMMoE(128, [f"t{i}" for i in range(6)])
    ret = model.forward(x)
    print(ret)





