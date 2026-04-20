from typing import Iterable

import torch
from torch import nn
from betterbole.models.backbone.common import build_expert_factory, build_gate_factory, build_tower_factory, to_dims
from betterbole.models.utils.container import MultiScenarioContainer, domain_select
from betterbole.models.utils.tests import dummy_input_multi_domain
import torch.nn.functional as F

class StarExpert(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation: str = 'relu', batch_norm=False):
        super().__init__()
        if isinstance(dims[0], Iterable):
            dims = dims[0]
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.batch_norm = batch_norm

        self.dropout_rate = dropout_rate
        self.W_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            W = nn.Parameter(torch.randn(in_dim, out_dim) / (in_dim ** 0.5))
            B = nn.Parameter(torch.zeros(out_dim))
            self.W_list.append(W)
            self.B_list.append(B)

    def merge_with(self, expert):
        W_list = [
            w1 * w2 for w1, w2 in
            zip(self.W_list, expert.W_list)
        ]
        B_list = [
            b1 + b2 for b1, b2 in
            zip(self.B_list, expert.B_list)
        ]
        return W_list, B_list


    @staticmethod
    def forward_with_params(W_list, B_list, x, activation='relu'):
        act_dict = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu
        }
        act_func = act_dict.get(activation.lower(), F.relu)
        W0, B0 = W_list[0], B_list[0]
        x = x @ W0 + B0
        for W, B in zip(W_list[1:], B_list[1:]):
            x = act_func(x)
            x = x @ W + B
        return x


class STAR(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        linear_stream = [emb_size, *expert_dims]
        rep_dim = expert_dims[-1]

        self.spe_experts = nn.ModuleList([StarExpert(*linear_stream) for  _ in range(num_domains)])
        self.sha_expert = StarExpert(*linear_stream)
        self.towers = MultiScenarioContainer(num_domains, build_tower_factory(rep_dim))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (max(1, emb_size // 2), emb_size, max(1, emb_size // 2), emb_size)

    def forward(self, x, domain_ids):
        merged_params = [self.sha_expert.merge_with(spe_expert) for spe_expert in self.spe_experts]
        experts_out = [StarExpert.forward_with_params(W_list, B_list, x) for W_list, B_list in merged_params]
        experts_out = torch.stack(experts_out, dim=1) # B K D

        experts_out = domain_select(experts_out, domain_ids) # B D
        towers_out = self.towers.forward(experts_out, domain_ids) # B D
        return towers_out.squeeze(-1)

class StarPle(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        linear_stream = [emb_size, *expert_dims]
        rep_dim = expert_dims[-1]

        ## STAR部分
        self.spe_experts = nn.ModuleList([StarExpert(*linear_stream) for _ in range(num_domains)])
        self.sha_expert = StarExpert(*linear_stream)

        ## PLE部分
        self.shared_gate_layer1 = build_gate_factory(emb_size, num_domains)()

        self.multi_experts_layer2 = MultiScenarioContainer(num_domains, build_expert_factory(rep_dim, expert_dims))
        self.shared_expert_layer2 = build_expert_factory(rep_dim, expert_dims)()
        self.multi_gates_layer2 = MultiScenarioContainer(num_domains, build_gate_factory(rep_dim, 1))
        self.towers = MultiScenarioContainer(num_domains, build_tower_factory(rep_dim))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (max(1, emb_size // 2), emb_size)

    def forward(self, x, domain_ids):
        merged_params = [self.sha_expert.merge_with(spe_expert) for spe_expert in self.spe_experts]
        experts_out = [StarExpert.forward_with_params(W_list, B_list, x) for W_list, B_list in merged_params]
        spe_interests = torch.stack(experts_out, dim=1)  # B K D
        spe_interest = domain_select(spe_interests, domain_ids)

        sha_gate = self.shared_gate_layer1(x).unsqueeze(-1) # B K 1
        sha_interest = torch.sum(sha_gate * spe_interests, dim=1)
        ## PLE部分
        spe_state = self.multi_experts_layer2(spe_interest, domain_ids)
        sha_state = self.shared_expert_layer2(sha_interest)
        spe_gates = self.multi_gates_layer2(spe_interest, domain_ids)
        output = spe_state * spe_gates + sha_state * (1 - spe_gates)
        return self.towers(output, domain_ids).squeeze(-1)

if __name__ == '__main__':
    model = StarPle(128, 6)
    x, domain_ids = dummy_input_multi_domain(6)
    ret = model.forward(x, domain_ids)

    print(ret.shape)
