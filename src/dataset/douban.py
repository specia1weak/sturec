from typing import List

from src.dataset.base import DatasetBase
from pathlib import Path
import polars as pl
"""
注意，这个数据集来自recbole-CDR
实际上USER文件完全相同
"""

def _drop_recbole_type(lf: pl.LazyFrame)->pl.LazyFrame:
    lf = lf.rename(lambda x: x.split(":")[0])
    return lf

class _DBDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Douban"
    BOOK_DIR = BASE_DIR / "DoubanBook"
    MOVIE_DIR = BASE_DIR / "DoubanMovie"
    MUSIC_DIR = BASE_DIR / "DoubanMusic"

    BOOK_INTER = BOOK_DIR / "DoubanBook.inter"
    BOOK_USER = BOOK_DIR / "DoubanBook.user"
    MOVIE_INTER = MOVIE_DIR / "DoubanMovie.inter"
    MOVIE_USER = MOVIE_DIR / "DoubanMovie.user"
    MUSIC_INTER = MUSIC_DIR / "DoubanMusic.inter"
    MUSIC_USER = MUSIC_DIR / "DoubanMusic.user"

    INTERS = {
        "book": BOOK_INTER,
        "movie": MOVIE_INTER,
        "music": MUSIC_INTER
    }
    USERS = {
        "book": BOOK_USER,
        "movie": MOVIE_USER,
        "music": MUSIC_USER
    }
    USER = BOOK_USER

    @property
    def MERGED_INTERS_LF(self)->pl.LazyFrame:
        lfs = []
        for name, inter in self.INTERS.items():
            lf = pl.scan_csv(inter, separator="\t")
            lf = lf.with_columns(
                pl.lit(name).alias("domain")
            )
            lfs.append(lf)
        lf_combined = pl.concat(lfs, how="vertical")
        return _drop_recbole_type(lf_combined)

    @property
    def ALL_INTERS_LF(self)-> List[pl.LazyFrame]:
        lfs = []
        for name, inter in self.INTERS.items():
            lf = pl.scan_csv(inter, separator="\t")
            lf = lf.with_columns(
                pl.lit(name).alias("domain")
            )
            lf = _drop_recbole_type(lf)
            lfs.append(lf)
        return lfs

    @property
    def USER_LF(self)->pl.LazyFrame:
        return _drop_recbole_type(pl.scan_csv(self.USER, separator="\t"))

DoubanDataset = _DBDS()
if __name__ == '__main__':
    import polars as pl
    from pprint import pprint
    from src.dataset.overview import get_general_info, get_head_info, get_group_stats
    df = DoubanDataset.MERGED_INTERS_LF.collect()
    ret = get_general_info(df)
    pprint(ret)

    ret = DoubanDataset.USER_LF.collect()
    pprint(get_general_info(ret))

    pprint(get_group_stats(df, "domain"))


