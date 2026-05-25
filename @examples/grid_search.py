# 定义搜索空间 (列表中包含你需要尝试的各种值)

if __name__ == '__main__':
    from betterbole.experiment import change_root_workdir
    change_root_workdir()
    from betterbole.experiment.engine import GridSearchEngine
    search_space = {
        "model": ["ppnet"],
        "seed": [2024, 2025],
        "device": ["cuda"],
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="@examples/kuairand-1k/kuairan1k.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)