from betterbole.models.backbone.common import build_expert_factory, build_gate_factory, build_tower_factory, to_dims
from betterbole.models.utils.general import MLP
from betterbole.models.utils.container import MultiScenarioContainer as MSC, domain_select
import torch
from torch import nn



class PLE(nn.Module):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__()
        expert_dims = to_dims(expert_dims, self.default_expert_dims(emb_size))
        rep_dim = expert_dims[-1]
        num_spe_experts = 1
        num_sha_experts = 1

        ## 第一层，共享兴趣由所有人共同设计，分兴趣很普通
        self.multi_experts_layer1 = nn.ModuleList(
            [MSC(num_domains, build_expert_factory(emb_size, expert_dims)) for _ in range(num_spe_experts)]
        )
        self.multi_gates_layer1 = MSC(num_domains, build_gate_factory(emb_size, num_spe_experts + num_sha_experts))
        self.shared_expert_layer1 =  nn.ModuleList(
            [build_expert_factory(emb_size, expert_dims)() for _ in range(num_sha_experts)]
        )
        self.shared_gates_layer1 = build_gate_factory(emb_size, num_spe_experts * num_domains + num_sha_experts)()
        ## 第二层，共享兴趣再被输入
        self.multi_experts_layer2 = nn.ModuleList(
            [MSC(num_domains, build_expert_factory(rep_dim, expert_dims)) for _ in range(num_spe_experts)]
        )
        self.multi_gates_layer2 = MSC(num_domains, build_gate_factory(rep_dim, num_spe_experts + num_sha_experts))
        self.shared_expert_layer2 = nn.ModuleList([
            build_expert_factory(rep_dim, expert_dims)() for _ in range(num_sha_experts)
        ])
        self.towers = MSC(num_domains, build_tower_factory(rep_dim))

    @staticmethod
    def default_expert_dims(emb_size):
        emb_size = max(1, int(emb_size))
        return (emb_size * 2, emb_size)

    def forward(self, x, domain_ids):
        """
        :param x: B, D
        """
        all_spe_feature = [spe_expert(x) for spe_expert in self.multi_experts_layer1] # List[B K D]
        spe_feature = [domain_select(one_expert_feature, domain_ids) for one_expert_feature in all_spe_feature] # List[B D]
        sha_feature = [sha_expert(x) for sha_expert in self.shared_expert_layer1] # List[B D]
        spe_gates = self.multi_gates_layer1(x, domain_ids).unsqueeze(-1) # B 2+2 1
        sha_gates = self.shared_gates_layer1(x).unsqueeze(-1)  # B K*2+2 1

        # 专属兴趣
        spe_sha_feature = torch.stack(spe_feature + sha_feature, dim=1) # B 2+2 D
        spe_interest = torch.sum(spe_sha_feature * spe_gates, dim=1) # B D

        # 共享兴趣
        all_feature = all_spe_feature + [sf.unsqueeze(dim=1) for sf in sha_feature] # [B K D, B K D, B 1 D, B 1 D]
        sha_interest = torch.sum(sha_gates * torch.cat(all_feature, dim=1), dim=1) # B D

        ## 第二层
        spe_state = [spe_expert(spe_interest, domain_ids) for spe_expert in self.multi_experts_layer2] # List[B D] = [B D, B D]

        sha_state = [sha_expert(sha_interest) for sha_expert in self.shared_expert_layer2] # List[B D] = [B D, B D]
        spe_gates = self.multi_gates_layer2(spe_interest, domain_ids).unsqueeze(-1) # B 2+2 1
        output = torch.sum(spe_gates * torch.stack(spe_state + sha_state, dim=1), dim=1)
        return self.towers(output, domain_ids).squeeze(-1)

# 版本1 低了两个点，难道是底层共享和PLE的设计理念冲突了？
class PLEVersion1(PLE):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__(emb_size, num_domains, expert_dims)
        self.pre_project = MLP(emb_size, emb_size)

    def forward(self, x, domain_ids):
        x = self.pre_project(x)
        return super().forward(x, domain_ids)

class PLEVersion2(PLE):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__(emb_size, num_domains, expert_dims)
        self.pre_project = MLP(emb_size, emb_size)

    def forward(self, x, domain_ids):
        scale = torch.sigmoid(self.pre_project(x))
        return super().forward(scale * x, domain_ids)

class PLEVersion3(PLE):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__(emb_size, num_domains, expert_dims)
        self.pre_project = nn.Linear(emb_size, emb_size)

    def forward(self, x, domain_ids):
        x = self.pre_project(x)
        return super().forward(x, domain_ids)

class PLEVersion4(PLE):
    def __init__(self, emb_size, num_domains, expert_dims=None):
        super().__init__(emb_size, num_domains, expert_dims)
        self.pre_project = MLP(emb_size, emb_size)

    def forward(self, x, domain_ids):
        residual = self.pre_project(x)
        return super().forward(x + residual, domain_ids)

if __name__ == '__main__':
    model = PLE(128, 5)

    from betterbole.utils.monitor import IndividualReLUMonitor
    from betterbole.models.utils.tests import dummy_input_multi_domain
    irm = IndividualReLUMonitor(model, 50)

    for _ in range(50):
        x, domain_ids = dummy_input_multi_domain(5, 10)
        y = model(x, domain_ids)
    print(y.shape)
    print(irm.get_layer_stats())
