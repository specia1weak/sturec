import ctypes
import math
import os
import time
import warnings
from enum import Enum
from typing import Union

from src.utils.singleton import Singleton


# windows 专用，不过脚本也只能在windows上跑了
if os.name != "nt":
    raise OSError
LOGGER = None
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


class Moment(Enum):
    CAPTURED = 1  # 截图
    PREPROCESSED = 2  # 图像预处理
    INFERRED = 3  # 预测
    EXECUTED = 4  # 执行
    TO_EXECUTE = 5  # 执行前


@Singleton
class TimeRecorder:
    def __init__(self):
        self._recorded_moment_dict = {}
        self._recorded_interval_dict = {}

    def record_moment(self, key):
        """
        :param key: 时间字典key
        :return: 上次记录的时间差
        """
        curr_time = self._get_curr_time()
        last_time = self._recorded_moment_dict.get(key, 0.)
        self._recorded_moment_dict[key] = curr_time
        interval = curr_time - last_time
        self._recorded_interval_dict[key] = interval
        return curr_time

    def get_record(self, key):
        return self._recorded_moment_dict.get(key, 0.)

    def calculate_same_key_record_interval(self, key):
        """
        记录上一次record和上上次record的时间差，注意必须要保证调用这个函数前使用了record
        """
        return self._recorded_interval_dict.get(key, 0.)

    def calculate_different_records_interval(self, key1, key2):
        return self.get_record(key2) - self.get_record(key1)

    def _get_curr_time(self):
        return time.perf_counter()

if __name__ == "__main__":
    @timer
    def test():
        very_high_precision_sleep(0.015)
    test()
    # for _ in range(100000):
    #     st = time.perf_counter()
    #     high_precision_sleep(0.002)
    #     print(time.perf_counter() - st)