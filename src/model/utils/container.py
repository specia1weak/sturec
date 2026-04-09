from typing import List, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
from src.model.utils.general import MLP

def domain_select(tensor, domain_ids):
    batch_size = tensor.size(0)
    batch_indices = torch.arange(batch_size, device=tensor.device)
    tensor = tensor[batch_indices, domain_ids]
    return tensor

class MultiScenarioCloneBase(nn.Module, ABC):
    """
    多场景分身模型抽象基类
    继承此类的子类只需实现 `build_single_domain_net` 即可自动获得多场景能力。
    """
    def __init__(self, num_domains: int):
        super().__init__()
        self.num_domains = num_domains
        self.domain_networks = nn.ModuleList([
            self.build_single_domain_net() for _ in range(num_domains)
        ])

    @abstractmethod
    def build_single_domain_net(self) -> nn.Module:
        """
        子类必须实现此方法。
        在这里 return 一个标准的单场景网络结构 (例如一个简单的 MLP)。
        """
        pass

    def forward(self, x: torch.Tensor, select_domain_ids: torch.Tensor=None):
        """
        x: [batch_size, feature_dim] 共享层输出的特征
        domain_ids: [batch_size] 场景标识 (值为 0 到 num_domains - 1)
        """
        all_results = []
        for i in range(self.num_domains):
            out_i = self.domain_networks[i](x)
            all_results.append(out_i)
        logits = torch.stack(all_results, dim=1)
        if select_domain_ids is not None:
            if select_domain_ids is not None:
                logits = domain_select(logits, select_domain_ids)
        return logits


class MultiScenarioContainer(nn.Module):
    def __init__(self, num_domains: int, network_factory: Callable[[], nn.Module]):
        super().__init__()
        self.num_domains = num_domains
        self.domain_networks = nn.ModuleList([
            network_factory() for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, select_domain_ids: torch.Tensor = None):
        all_results = []
        for i in range(self.num_domains):
            out_i = self.domain_networks[i](x)
            all_results.append(out_i)

        logits = torch.stack(all_results, dim=1)
        if select_domain_ids is not None:
            if select_domain_ids is not None:
                logits = domain_select(logits, select_domain_ids)
        return logits

    def __getitem__(self, idx):
        return self.domain_networks[idx]


class MultiTaskContainer(nn.Module):
    """
    多任务网络容器。
    不同于多场景的“按需选择”，多任务需要对输入特征进行“全量分发与收集”。
    """
    def __init__(self, task_names: List[str], network_factory: Callable[[], nn.Module]):
        """
        :param task_names: 任务名称列表，例如 ['ctr', 'cvr']
        :param network_factory: 实例化单任务网络的工厂函数
        """
        super().__init__()
        self.task_names = task_names
        self.task_networks = nn.ModuleDict({
            task: network_factory() for task in task_names
        })

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: 底层提取出来的共享特征 [Batch, Feature_Dim]
        kwargs: 透传参数（例如可以把 domain_ids 透传给内部的多场景网络）
        """
        task_logits = {}
        for task_name, task_net in self.task_networks.items():
            task_logits[task_name] = task_net(x, **kwargs)
        return task_logits

    def __getitem__(self, task_name):
        return self.task_networks[task_name]



if __name__ == '__main__':
    class PLE(nn.Module):
        def __init__(self, num_domains):
            super().__init__()
            ## 第一层，共享兴趣由所有人共同设计，分兴趣很普通
            self.multi_experts_layer1 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: MLP(128, 64, 128)
            )
            # 2. 直接传入 lambda 函数来实例化多场景门控网络
            self.multi_gates_layer1 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )
            self.shared_expert_layer1 = MLP(128, 64, 128)
            self.shared_gates_layer1 = nn.Sequential(
                MLP(128, num_domains + 1),
                nn.Softmax(dim=-1)
            )

            ## 第二层，共享兴趣再被输入
            self.multi_experts_layer2 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: MLP(128, 64, 128)
            )
            # 2. 直接传入 lambda 函数来实例化多场景门控网络
            self.multi_gates_layer2 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )
            self.shared_expert_layer2 = MLP(128, 64, 128)
            self.towers = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )

        def forward(self, x, domain_ids):
            """
            :param x: B, D
            """
            all_feature = self.multi_experts_layer1(x)
            spe_feature = domain_select(all_feature, domain_ids)
            sha_feature = self.shared_expert_layer1(x)
            spe_gates = self.multi_gates_layer1(x, domain_ids)
            sha_gates = self.shared_gates_layer1(x).unsqueeze(-1) # B K+1 1

            spe_interest = spe_feature * spe_gates + sha_feature * (1 - spe_gates)
            weighted_spe_feature = sha_gates * torch.cat([all_feature, sha_feature.unsqueeze(1)], dim=1)
            sha_interest = torch.sum(weighted_spe_feature, dim=1)

            ## 第二层
            spe_state = self.multi_experts_layer2(spe_interest, domain_ids)
            sha_state = self.shared_expert_layer2(sha_interest)
            spe_gates = self.multi_gates_layer2(spe_interest, domain_ids)
            return self.towers(spe_state * spe_gates + sha_state * (1 - spe_gates), domain_ids).squeeze()


    # 设置超参
    BATCH_SIZE = 4
    FEATURE_DIM = 128
    NUM_DOMAINS = 3  # 假设有3个不同的场景 (0, 1, 2)

    # 初始化模型
    model = PLE(num_domains=NUM_DOMAINS)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    domain_ids = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    print("=== 输入信息 ===")
    print(f"输入 x 形状: {x.shape}")
    print(f"输入 domain_ids: {domain_ids}\n")

    # 执行前向传播
    final_out = model(x, domain_ids)

    print("=== 提取结果证明 ===")
    print(f"模型最终输出 形状: {final_out.shape}")

    # 验证最终形状是否符合预期 [Batch, Feature_Dim]
    assert final_out.shape == (BATCH_SIZE, FEATURE_DIM), "最终输出维度错误！"
    print("\n✅ Forward pass 成功，没有发生维度崩溃！")
