from functools import cached_property
from pathlib import Path

import pandas as pd
import polars as pl
from betterbole.datasets.base import DatasetBase

class _MLDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR /"MovieLens"
    ML_1M_DIR = BASE_DIR / "ml-1m"
    USER_FEATURES = ML_1M_DIR / "users.dat"
    MOVIES_FEATURES = ML_1M_DIR / "movies.dat"
    RATINGS_DATA = ML_1M_DIR / "ratings.dat"

    @cached_property  # 改用 cached_property
    def ITEM_FEATURES_DF(self):
        return pd.read_csv(
            self.MOVIES_FEATURES,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

    @cached_property
    def USER_FEATURES_DF(self):
        return pd.read_csv(
            self.USER_FEATURES,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip-code']
        )

    @cached_property
    def INTERACTION_DF(self):
        return pd.read_csv(
            self.RATINGS_DATA,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

MovieLensDataset = _MLDS()
# 使用方法：
if __name__ == "__main__":
    from betterbole.datasets.overview import get_general_info, get_head_info, get_group_stats
    import pprint
    user_df = pl.from_pandas(MovieLensDataset.USER_FEATURES_DF)
    item_df = pl.from_pandas(MovieLensDataset.ITEM_FEATURES_DF)
    inter_df = pl.from_pandas(MovieLensDataset.INTERACTION_DF)
    info = get_general_info(user_df)
    pprint.pprint(info, depth=4, compact=False)
    info = get_general_info(item_df)
    pprint.pprint(info, depth=4, compact=False)
    info = get_general_info(inter_df)
    pprint.pprint(info, depth=4, compact=False)
