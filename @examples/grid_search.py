# 定义搜索空间 (列表中包含你需要尝试的各种值)

if __name__ == '__main__':
    from betterbole.experiment import change_root_workdir
    change_root_workdir()
    from betterbole.experiment.engine import GridSearchEngine
    search_space = {
        "model": ["ple_shavq_v3"],
        "seed": [2026],
        "weight_decay": [1e-6],
        "device": ["cuda"],
        "ple_shavq_domain_balanced_ema": [True],
        "ple_balanced_balanced_domain_adv_weight": [0.0],
        "ple_shavq_residual_scale": [0.15, 0.25],
        "ple_shavq_commitment_weight": [0.1, 0.2],
        "ple_shavq_codebook_size": [64],
        "ple_balanced_common_probe_weight": [0.005, 0.01],
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="@examples/kuairand-1k/kuairan1k.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)
