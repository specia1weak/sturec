import os
import psutil
# 指定一个本地锁文件用于传递 PID
from pathlib import Path
LOCK_FILE = Path(__file__).parent / "task_chain.lock"
"""
潜在的小隐患：PID 重用（极低概率）
虽然代码很稳健，但在 Windows 上存在一个理论上的极端情况：PID 复用。

Windows 的进程 ID 是循环使用的。如果：

前者（PID: 1001）意外崩溃。

在后者启动前的千万分之一秒内，系统刚好把 PID 1001 分配给了一个全新的、无关的长期运行程序（比如你的浏览器）。

后者读取到了 1001，发现这个进程竟然还在跑，它就会一直等下去。
"""
def auto_queue():
    my_pid = os.getpid()
    wait_pid = None

    # 1. 尝试读取上一个任务的 PID
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                pid_str = f.read().strip()
                if pid_str.isdigit():
                    wait_pid = int(pid_str)
        except Exception:
            pass

    # 2. 立刻把自己的 PID 写进去，让后面追加的任务等自己
    with open(LOCK_FILE, "w") as f:
        f.write(str(my_pid))

    # 3. 如果发现有前置任务，进入阻塞等待
    if wait_pid and wait_pid != my_pid:
        try:
            p = psutil.Process(wait_pid)
            if p.is_running():
                print(f"[排队系统] 已挂起。正在等待前置任务 (PID: {wait_pid}) 完成...")
                p.wait()  # 进程在这里完全停住，不消耗 CPU
        except psutil.NoSuchProcess:
            # 说明上一个任务刚好在这极短的时间内跑完，或者 PID 已经失效
            pass

    print(f"\n[排队系统] 前置任务已清空，当前任务 (PID: {my_pid}) 开始执行！")


if __name__ == "__main__":
    # 在所有业务逻辑开始前调用
    auto_queue()

    # ==========================================
    # 下面写你经常修改的业务代码
    # ==========================================
    import time

    print("代码版本 A 正在运行...")
    time.sleep(10)  # 模拟长耗时任务
    print("代码版本 A 运行结束。")