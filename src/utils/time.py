import ctypes
import math
import os
import time
import warnings
from enum import Enum
from typing import Union

# windows 专用，不过脚本也只能在windows上跑了
if os.name != "nt":
    raise OSError
LOGGER = print
_SPARE_TIME = 0.003 # s
_BIAS_TIME = 0.002

DEFAULT_STRICT_PRECISION = False

def _windll_sleep(sleep_second):
    ctypes.windll.winmm.timeBeginPeriod(1)
    time.sleep(max(0, sleep_second - _BIAS_TIME / 2))
    ctypes.windll.winmm.timeEndPeriod(1)


def very_high_precision_sleep(delay_time):
    entry_time = time.perf_counter()
    _windll_sleep(max(0, delay_time - _SPARE_TIME))
    left_time = delay_time - (time.perf_counter() - entry_time)
    target = time.perf_counter() + left_time
    while time.perf_counter() < target:
        time.sleep(0)

def high_precision_sleep(delay_time, strict_high: Union[None, bool]=None):
    """ Function to provide accurate time delay in millisecond
    """
    st_moment = time.perf_counter()
    if strict_high is None:
        strict_high = DEFAULT_STRICT_PRECISION
    if not strict_high:
        _windll_sleep(delay_time)
    else:
        very_high_precision_sleep(delay_time)
    sleep_time = time.perf_counter() - st_moment
    if abs(sleep_time - delay_time) > _SPARE_TIME:
        warnings.warn(f"【警告】:sleep唤醒过晚，偏差{sleep_time - delay_time:.4f}")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.perf_counter()  # 记录函数结束执行的时间
        if LOGGER is not None:
            print(f"【函数计时】:{func.__name__} 使用 {end_time - start_time:.4f} s")
        return result

    return wrapper


def set_min_time(min_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            exec_time = time.perf_counter() - start_time
            wait_time = max(0, min_time - exec_time)
            if wait_time > 0:
                if LOGGER is not None:
                    print(f"【耗时对齐】:等待{wait_time:.6f}秒以满足最小时间要求")
                high_precision_sleep(wait_time)
            return result

        return wrapper

    return decorator


import time
import functools
from collections import defaultdict


class NamedTimer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NamedTimer, cls).__new__(cls)
            cls._instance._records = defaultdict(float)
            cls._instance._counts = defaultdict(int)  # 新增：记录调用次数
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

    # --- 核心：装饰器功能 ---
    def collect(self, name=None):
        """
        装饰器用法：
        @timer.collect()
        def my_func(): ...
        """

        def decorator(func):
            # 如果没传 name，默认用函数名
            record_name = name if name else func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_record(record_name)
                result = func(*args, **kwargs)
                self.stop_record(record_name)
                return result

            return wrapper

        return decorator

    # --- 之前的 Context Manager 支持 ---
    def __call__(self, name):
        self._temp_name = name
        return self

    def __enter__(self):
        self.start_record(self._temp_name)
        return self

    def __exit__(self, *args):
        self.stop_record(self._temp_name)

    def report(self):
        print(f"\n{'[ Timer Report ]':^40}")
        print(f"{'Name':<20} | {'Total':>8} | {'Avg':>8} | {'Calls':>5}")
        print("-" * 45)
        for name in self._records:
            total = self._records[name]
            count = self._counts[name]
            avg = total / count if count > 0 else 0
            print(f"{name:<20} | {total:>7.4f}s | {avg:>7.4f}s | {count:>5}")