import enum
from pathlib import Path
from typing import Iterable

import pandas as pd

class CSVType(enum.Enum):
    item = "item"
    inter = "inter"
    user = "user"


def _infer_iterable_type(it: Iterable):
    # 惰性容器先转成迭代器，避免一次性把生成器耗尽
    it = iter(it)

    # 找第一个非 None 的元素
    for x in it:
        if x is None:  # 把 None 当成“未知”继续找
            continue
        if isinstance(x, bool):  # bool 是 int 的子类，要先排掉
            return 'bool'
        if isinstance(x, int):
            return 'int'
        if isinstance(x, str):
            return 'str'
        return type(x).__name__  # 其他类型直接给类名
    return None

def load_recbole(file_path, usecols=None):
    if isinstance(usecols, Iterable) and _infer_iterable_type(usecols) == "str":
        usecols = tuple(usecols)
        return pd.read_csv(file_path, usecols=lambda c: c.split(":")[0] in usecols, sep='\t')
    else:
        return pd.read_csv(file_path, usecols=usecols, sep='\t')


class RecboleLoader:
    def __init__(self, file_path, usecols=None):
        self.file_path = Path(file_path)
        self.file_type = self.file_path.suffix.lower()[1:]
        self.df = self.load_df(usecols)

    def load_df(self, usecols=None):
        return load_recbole(self.file_path, usecols=usecols)


if __name__ == '__main__':
    from src.dataset.amaz import AMAZON_MOVIES_INTER, AMAZON_MOVIES_ITEM
    df = RecboleLoader(AMAZON_MOVIES_INTER, usecols=["user_id", "item_id", "rating"]).df
    print(df.head(5))
    df = RecboleLoader(AMAZON_MOVIES_ITEM, usecols=["item_id", "title"]).df
    print(df.head(5))