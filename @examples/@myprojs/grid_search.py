from betterbole.experiment.engine import GridSearchEngine
if __name__ == '__main__':
    # 定义搜索空间 (列表中包含你需要尝试的各种值)
    search_space = {
        "backbone": ["star", "ple"],
        "seed": [2024, 2025, 2026],
        "device": ["cuda"]
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="kuairan1k.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)