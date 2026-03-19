from pathlib import Path
from typing import Union, List, Callable

from sklearn.preprocessing import LabelEncoder

from .constents import RAW_DATASET_BASE_DIR, BOLE_DATASET_BASE_DIR
CROSS_DOMAIN_AMAZON =  RAW_DATASET_BASE_DIR / "Amazon"
AMAZON_BOOKS_DIR =  CROSS_DOMAIN_AMAZON / "AmazonBooks"
AMAZON_BOOKS_INTER =  AMAZON_BOOKS_DIR / "AmazonBooks.inter"
AMAZON_BOOKS_ITEM =  AMAZON_BOOKS_DIR / "AmazonBooks.item"

AMAZON_MOVIES_DIR =  CROSS_DOMAIN_AMAZON / "AmazonMov"
AMAZON_MOVIES_INTER =  AMAZON_MOVIES_DIR / "AmazonMov.inter"
AMAZON_MOVIES_ITEM =  AMAZON_MOVIES_DIR / "AmazonMov.item"
import pandas as pd
# user_id:token	item_id:token	rating:float	timestamp:float
# item_id:token	title:token	price:float	brand:token

from .base import load_recbole, CSVType


class AmazonCSVLoader:
    def __init__(self, domain_dir, item_usecols=None, inter_usecols=None, user_usecols=None):
        self.df_item = None
        self.df_user = None
        self.df_inter = None
        self.domain_name = Path(domain_dir).name
        self.field_types = {}

        usecols_map = {
            CSVType.item.name: item_usecols,
            CSVType.inter.name: inter_usecols,
            CSVType.user.name: user_usecols
        }

        for csv_type_enum in CSVType:
            csv_type = csv_type_enum.name
            file_path = Path(domain_dir) / f"{self.domain_name}.{csv_type}"

            if file_path.exists():
                # 1. 原始加载
                df_raw = load_recbole(file_path, usecols=usecols_map[csv_type])

                # 2. 清洗并记忆类型
                df_clean = self._strip_and_record_types(df_raw)

                setattr(self, f"df_{csv_type}", df_clean)
            else:
                setattr(self, f"df_{csv_type}", None)


    def _strip_and_record_types(self, df):
        """
        剥离列名后缀，并记录到 self.field_types 中
        """
        new_columns = []
        for col in df.columns:
            if ":" in col:
                # 拆分 'rating:float' -> name='rating', verify_type='float'
                name, raw_type = col.split(":", 1)

                # 记录类型 (如果两个表都有 user_id:token，update 会覆盖，但通常类型是一致的)
                self.field_types[name] = raw_type
                new_columns.append(name)
            else:
                # 如果原文件里没有写类型 (比如 csv 没头)，默认设为 token 或 ignore
                name = col
                if name not in self.field_types:
                    self.field_types[name] = 'token'  # 默认兜底
                new_columns.append(name)

        df.columns = new_columns
        return df

    def _clean_columns(self, df):
        """
        将 "user_id:token" 重命名为 "user_id"，并返回类型映射
        """
        new_columns = []
        type_map = {}
        for col in df.columns:
            if ":" in col:
                name, col_type = col.split(":", 1)
                new_columns.append(name)
                type_map[name] = col_type
            else:
                new_columns.append(col)
                # 如果没有类型后缀，默认视为 token 或不做处理，视情况而定
                type_map[col] = 'token'

        df.columns = new_columns
        return df, type_map

    def _iterative_k_core(self, df: pd.DataFrame, k=5, user_col='user_id', item_col='item_id'):
        """
        现在这个函数变得非常清爽，不需要映射表了
        """
        # 确保列存在，防止报错
        if user_col not in df.columns or item_col not in df.columns:
            return df

        while True:
            start_len = len(df)

            # 1. 过滤用户
            user_counts = df[user_col].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df[user_col].isin(valid_users)]

            # 2. 过滤物品
            item_counts = df[item_col].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df[item_col].isin(valid_items)]

            if len(df) == start_len:
                break

        print(f"[{self.domain_name}] Finished {k}-core. Size: {start_len} -> {len(df)}")
        return df

    def filter_core(self, core=5):
        if self.df_inter is None:
            return

        # 1. 过滤交互表 (直接使用纯净的 user_id, item_id)
        self.df_inter = self._iterative_k_core(self.df_inter, k=core)

        # 2. 同步过滤 item 表
        if self.df_item is not None:
            # 直接用 item_id，不需要查找映射
            valid_items = self.df_inter['item_id'].unique()
            self.df_item = self.df_item[self.df_item['item_id'].isin(valid_items)].copy()

class AmazonBooksLoader(AmazonCSVLoader):
    def __init__(self):
        super().__init__(domain_dir=AMAZON_BOOKS_DIR,
                         inter_usecols=["user_id", "item_id", "rating", "timestamp"],
                         item_usecols=["item_id", "title"])
        self.df_item: pd.DataFrame = self.df_item  # super init 得到的属性
        self.df_inter: pd.DataFrame = self.df_inter  # super init 得到的属性


class AmazonMoviesLoader(AmazonCSVLoader):
    def __init__(self):
        super().__init__(domain_dir=AMAZON_MOVIES_DIR,
                         inter_usecols=["user_id", "item_id", "rating", "timestamp"],
                         item_usecols=["item_id", "title"])
        self.df_item: pd.DataFrame = self.df_item  # super init 得到的属性
        self.df_inter: pd.DataFrame = self.df_inter  # super init 得到的属性


class AmazonUserLoader:
    def __init__(self, df_inter: pd.DataFrame):
        """
        :param df_inter: 必须包含清洗后的列 ['user_id', 'item_id', 'domain', 'rating', 'timestamp']
        """
        self.history_dict = {}
        self._build_index(df_inter)

    def _build_index(self, df):
        print("Indexing user history...")
        # 1. 确保按时间排序，这对序列推荐是必须的
        #    注意：这里假设传入的 user_id 已经是 Int 类型
        sorted_df = df.sort_values(by=['user_id', 'timestamp'])

        # 2. 为了支持复杂的过滤，我们将多列打包成 tuple
        #    结构: (item_id, domain, rating, timestamp)
        #    这样在内存中比 dict 更省空间，且比 DataFrame 查询快得多
        sorted_df['meta_tuple'] = list(zip(
            sorted_df['item_id'],
            sorted_df['domain'],
            sorted_df['rating'],
            sorted_df['timestamp']
        ))

        # 3. 聚合
        self.history_dict = sorted_df.groupby('user_id')['meta_tuple'].apply(list).to_dict()
        print(f"User history indexed for {len(self.history_dict)} users.")

    def get_history(self,
                    user_id: int,
                    domains: Union[str, List[str]] = None,
                    filter_func: Callable = None,
                    return_type: str = 'item_id') -> list:
        """
        获取用户历史序列的核心函数

        :param user_id: 用户的 Int ID
        :param domains: 过滤域，例如 'book' 或 ['book', 'movie']。None 表示不过滤。
        :param filter_func: 自定义过滤函数，接收参数为 (item_id, domain, rating, timestamp)，返回 Bool
        :param return_type: 返回数据的格式。
               - 'item_id': 只返回 [item_id1, item_id2...] (常用)
               - 'full': 返回 [(item_id1, dom1, rat1, ts1), ...] (用于调试或复杂构造)
        """
        if user_id not in self.history_dict:
            return []

        raw_seq = self.history_dict[user_id]

        # 预处理 domains 参数为 set 加速查找
        if domains:
            if isinstance(domains, str):
                target_domains = {domains}
            else:
                target_domains = set(domains)
        else:
            target_domains = None

        result_seq = []

        for item_tuple in raw_seq:
            # item_tuple 解包: 0:id, 1:domain, 2:rating, 3:timestamp
            current_domain = item_tuple[1]

            # 1. Domain 过滤
            if target_domains and current_domain not in target_domains:
                continue

            # 2. 自定义函数过滤
            #    filter_func 接收完整 tuple，用户可以写 lambda t: t[2] > 3 (评分大于3)
            if filter_func and not filter_func(item_tuple):
                continue

            # 3. 组装返回结果
            if return_type == 'item_id':
                result_seq.append(item_tuple[0])
            else:
                result_seq.append(item_tuple)

        return result_seq

    def get_interaction_count(self, user_id):
        return len(self.history_dict.get(user_id, []))


class AmazonCrossDomainLoader:
    def __init__(self, filter_k_core=5):
        self.n_items = None
        self.n_users = None
        self.book_loader = AmazonBooksLoader()  # 继承上面的修改版
        self.movies_loader = AmazonMoviesLoader()

        # 注意：k-core 最好在各自域内部做，还是合并后做？
        # 通常 RecBole 逻辑是先分别过滤，再合并。
        self.merged_field_types = {}
        self.merged_field_types.update(self.book_loader.field_types)
        self.merged_field_types.update(self.movies_loader.field_types)
        if filter_k_core:
            self.book_loader.filter_core(filter_k_core)
            self.movies_loader.filter_core(filter_k_core)

        self.union_df_inter, self.union_df_item = self.merge_and_encode()
        self.user_loader = AmazonUserLoader(self.union_df_inter)

    def merge_and_encode(self):
        # 此时 df_inter_book 的列已经是 [user_id, item_id, rating, ...] 没有 :token
        df_inter_book = self.book_loader.df_inter.copy()
        df_item_book = self.book_loader.df_item.copy()

        df_inter_movie = self.movies_loader.df_inter.copy()
        df_item_movie = self.movies_loader.df_item.copy()

        # --- 处理 ID 冲突 (加上前缀) ---
        # 这样写比之前处理 :token 要安全得多
        df_inter_book['item_id'] = 'book_' + df_inter_book['item_id'].astype(str)
        df_inter_movie['item_id'] = 'movie_' + df_inter_movie['item_id'].astype(str)

        df_item_book['item_id'] = 'book_' + df_item_book['item_id'].astype(str)
        df_item_movie['item_id'] = 'movie_' + df_item_movie['item_id'].astype(str)

        df_inter_book['domain'] = 'book'
        df_inter_movie['domain'] = 'movie'

        # --- 合并 ---
        union_inter = pd.concat([df_inter_book, df_inter_movie], ignore_index=True)
        union_item = pd.concat([df_item_book, df_item_movie], ignore_index=True)

        # --- Label Encoding ---
        print("Encoding User IDs...")
        user_le = LabelEncoder()

        # [修改点 1] 备份 User ID Token
        union_inter['user_id_token'] = union_inter['user_id']
        union_inter['user_id'] = user_le.fit_transform(union_inter['user_id']) + 1

        print("Encoding Item IDs...")
        item_le = LabelEncoder()

        # [修改点 2] 备份 Item ID Token (关键！你之前缺了这个)
        # 这时候的 item_id 已经是 'book_1001', 'movie_2002' 这种格式了
        # 我们必须把它存下来，导出的时候要用
        union_inter['item_id_token'] = union_inter['item_id']
        union_item['item_id_token'] = union_item['item_id']

        # 获取所有可能的 item id 并编码
        all_items = pd.concat([union_inter['item_id'], union_item['item_id']]).unique()
        item_le.fit(all_items)

        # 转换成数字 (为了内存计算方便)
        union_inter['item_id'] = item_le.transform(union_inter['item_id']) + 1
        union_item['item_id'] = item_le.transform(union_item['item_id']) + 1
        self.n_users = len(user_le.classes_) + 1
        self.n_items = len(item_le.classes_) + 1
        self.merged_field_types['domain'] = 'token'
        return union_inter, union_item

    def export_to_recbole_format(self, df_type='inter', to_csv_path=None):
        if df_type == 'inter':
            df = self.union_df_inter.copy()
            if 'rating' in df.columns:
                print(f"[Export] 正在执行过滤策略 (保留 Rating >= 4)...")
                original_len = len(df)
                df = df[df['rating'] >= 4].copy()
                df['label'] = 1.0
                df.drop(columns=['rating'], inplace=True)
                print(f"[Export] 过滤完成: {original_len} -> {len(df)} (保留率: {len(df) / original_len:.2%})")
                # 4. 更新类型映射 (让表头变成 label:float 而不是 rating:float)
                self.merged_field_types['label'] = 'float'
                # 如果字典里有 rating，把它删掉，防止报错或多余映射
                if 'rating' in self.merged_field_types:
                    del self.merged_field_types['rating']
        elif df_type == 'item':
            df = self.union_df_item.copy()
        else:
            raise ValueError("Unknown df_type")

        # [修改点 3] 还原数据：把 Token 列的值 赋给 ID 列
        # 这样导出时，user_id 就是 'A10...' 而不是 614
        # item_id 就是 'book_10...' 而不是 0
        if 'user_id_token' in df.columns:
            df['user_id'] = df['user_id_token']
            df.drop(columns=['user_id_token'], inplace=True)  # 删掉冗余列

        if 'item_id_token' in df.columns:
            df['item_id'] = df['item_id_token']
            df.drop(columns=['item_id_token'], inplace=True)  # 删掉冗余列

        # --- 以下重命名逻辑保持不变 ---
        rename_map = {}
        for col in df.columns:
            # 查找类型，找不到默认用 token
            col_type = self.merged_field_types.get(col, 'token')

            # 因为上面已经 drop 掉了 _token 列，这里其实不需要额外的 if col.endswith 检查了
            # 但保留着也无妨
            if col.endswith('_token'):
                continue

            rename_map[col] = f"{col}:{col_type}"

        # 执行重命名
        df_export = df.rename(columns=rename_map)

        if to_csv_path:
            # 建议用 with_suffix
            save_path = Path(to_csv_path).with_suffix(f".{df_type}")
            df_export.to_csv(save_path, sep='\t', index=False)
            print(f"已导出: {save_path}")

        return df_export