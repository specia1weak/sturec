import ctypes
import math
import os
import time
import warnings
from enum import Enum
from typing import Union

LOGGER = print

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.perf_counter()  # 记录函数结束执行的时间
        if LOGGER is not None:
            print(f"【函数计时】:{func.__name__} 使用 {end_time - start_time:.4f} s")
        return result

    return wrapper


import functools
import contextlib  # <--- 新增导入
from collections import defaultdict

class NamedTimer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NamedTimer, cls).__new__(cls)
            cls._instance._records = defaultdict(float)
            cls._instance._counts = defaultdict(int)
            cls._instance._start_times = {}
        return cls._instance

    def start_record(self, name):
        self._start_times[name] = time.perf_counter()

    def stop_record(self, name):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times.pop(name)
            self._records[name] += elapsed
            self._counts[name] += 1
            return elapsed
        return 0

    def loop_flag(self, name):
        self.stop_record(name)
        self.start_record(name)

    def collect(self, name=None):
        def decorator(func):
            record_name = name if name else func.__name__
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_record(record_name)
                result = func(*args, **kwargs)
                self.stop_record(record_name)
                return result
            return wrapper
        return decorator

    # --- 修复点：使用 contextmanager 替代原来的 __call__, __enter__, __exit__ ---
    @contextlib.contextmanager
    def __call__(self, name):
        self.start_record(name)
        try:
            yield self  # 允许 with ntr("name") as t: 的语法
        finally:
            self.stop_record(name)

    def report(self):
        print(f"\n{'[ Timer Report ]':^40}")
        print(f"{'Name':<20} | {'Total':>8} | {'Avg':>8} | {'Calls':>5}")
        print("-" * 45)
        for name in self._records:
            total = self._records[name]
            count = self._counts[name]
            avg = total / count if count > 0 else 0
            print(f"{name:<20} | {total:>7.4f}s | {avg:>7.4f}s | {count:>5}")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class CudaNamedTimer(NamedTimer):
        def start_record(self, name):
            # 只有在 GPU 可用时才同步，避免在只有 CPU 的 torch 环境下报错
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            super().start_record(name)

        def stop_record(self, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return super().stop_record(name)
else:
    # 如果没有 torch，CudaNamedTimer 就退化为普通的 NamedTimer
    # 这样业务代码里写 CudaNamedTimer() 也不会崩，只是没有同步功能
    class CudaNamedTimer(NamedTimer):
        pass