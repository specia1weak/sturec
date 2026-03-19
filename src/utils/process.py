from typing import Iterable

import psutil
ABOVE_NORMAL_PRIORITY_CLASS = psutil.ABOVE_NORMAL_PRIORITY_CLASS
NORMAL_PRIORITY_CLASS = psutil.NORMAL_PRIORITY_CLASS
HIGH_PRIORITY_CLASS = psutil.HIGH_PRIORITY_CLASS
LARGE_CORE_CPUS = tuple([i for i in range(16)])
SMALL_CORE_CPUS = tuple([i for i in range(16, 24)])
def set_priority(ps_priority=psutil.HIGH_PRIORITY_CLASS):
    import os
    # 获取当前进程
    current_process = psutil.Process(os.getpid())
    # 设置优先级（Windows 可用值：psutil.HIGH_PRIORITY_CLASS）
    current_process.nice(ps_priority)  # 等同于 "high priority"
    # 查看当前优先级
    print(current_process.nice())

def get_cpu_load_rank():
    per_cpu_percent = psutil.cpu_percent(interval=10, percpu=True)
    load_rank = sorted(enumerate(per_cpu_percent), key=lambda x: -x[-1])
    return load_rank

def set_affinity(pid, cpus):
    process = psutil.Process(pid)
    process.cpu_affinity(cpus)

def get_affinity(pid):
    process = psutil.Process(pid)
    affinity = process.cpu_affinity()
    print(f"进程可以运行的CPU核心: {affinity}")

import psutil
def get_idle_cpus(nums, groups=1, exclude_cpus: Iterable=None):
    if exclude_cpus is None:
        exclude_cpus = tuple()
    logical_cpu_count = psutil.cpu_count(logical=True)
    if nums * groups > logical_cpu_count:
        raise ValueError
    load_rank = get_cpu_load_rank()
    print("cpu负载", load_rank)
    idle_cpus = [x[0] for x in load_rank if x[0] not in exclude_cpus]
    idle_cpus = idle_cpus[-nums*groups: ]
    return [idle_cpus[i:i + nums] for i in range(0, len(idle_cpus), nums)]
