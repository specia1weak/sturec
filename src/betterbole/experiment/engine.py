import itertools
import subprocess
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any


class GridSearchEngine:
    def __init__(self, script_path: str,):
        self.script_path = script_path
        self.base_python = sys.executable

    def make_grid(self, param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """将参数空间字典展开为笛卡尔积列表"""
        keys = param_space.keys()
        values = param_space.values()
        combinations = itertools.product(*values)
        return [dict(zip(keys, combo)) for combo in combinations]

    def build_experiment_name(self, params: Dict[str, Any]) -> str:
        """根据当前参数组合动态生成实验名"""
        parts = []
        for k, v in params.items():
            if k == "experiment_name":
                continue
            # 简化浮点数等长格式
            val_str = str(v).replace(".", "_") if isinstance(v, float) else str(v)
            parts.append(f"{k}{val_str}")
        return "-".join(parts) if parts else "baseline"

    def worker(self, params: Dict[str, Any], gpu_id: int, log_dir: str):
        """单个实验的执行工作流"""
        # 1. 构建实验名并注入参数
        exp_name = self.build_experiment_name(params)
        if "experiment_name" not in params:
            params["experiment_name"] = exp_name
        cmd = [self.base_python, self.script_path]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{exp_name}.log")

        print(f"🚀 [Start] GPU:{gpu_id} | Exp:{exp_name} | Cmd: {' '.join(cmd)}")
        start_time = time.time()

        try:
            with open(log_path, "w") as log_file:
                process = subprocess.run(
                    cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
            status = "✅ Success" if process.returncode == 0 else "❌ Failed"
        except Exception as e:
            status = f"💥 Error: {e}"

        cost_time = time.time() - start_time
        print(f"{status} [{cost_time:.1f}s] | Exp:{exp_name}")
        return exp_name, status

    def run(self, param_space: Dict[str, List[Any]], available_gpus: List[int], log_dir: str = "./logs"):
        tasks = self.make_grid(param_space)
        print(f"📊 发现 {len(tasks)} 个实验组合，可用 GPU: {available_gpus}")
        max_workers = len(available_gpus) if available_gpus else 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, task_params in enumerate(tasks):
                # 轮询分配 GPU
                gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
                future = executor.submit(self.worker, task_params, gpu_id, log_dir)
                futures.append(future)
            for future in as_completed(futures):
                future.result()


if __name__ == '__main__':
    # 定义搜索空间 (列表中包含你需要尝试的各种值)
    search_space = {
        "model": ["star", "ple"],
        "seed": [2024, 2025, 2026],
        "device": ["cuda"]
    }

    # 假设你有一台服务器，有两张卡可以跑实验
    gpus_to_use = [0]
    engine = GridSearchEngine(script_path="run.py")
    engine.run(param_space=search_space, available_gpus=gpus_to_use)