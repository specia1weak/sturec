import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from src.utils.time import timer


class InterQuery:
    def __init__(self, inter_path, max_len=10):
        self.max_len = max_len
        self.df = self._prepare_df(inter_path)

    def _prepare_df(self, inter_path):
        df = pd.read_csv(inter_path, sep='\t')
        rename_map = {
            col: col.split(":")[0] for col in df.columns
        }
        df = df.rename(columns=rename_map)
        df = df.sort_values("timestamp").reset_index(drop=True)

        def get_user_history_series(user_df, max_len=self.max_len):
            items = user_df['item_id'].tolist()
            ratings = user_df['rating'].tolist()

            histories = []
            current_hist = []

            for item, rating in zip(items, ratings):
                histories.append(list(current_hist[-max_len:]))
                if rating >= 3:
                    current_hist.append(item)
            return pd.Series(histories, index=user_df.index)

        # 2. 直接赋值给原表的新列
        # 加了 include_groups=False，明确告诉 Pandas 里面不需要 user_id，消除警告
        df['history'] = df.groupby("user_id", group_keys=False).apply(
            get_user_history_series,
            include_groups=False
        )
        df = df.set_index(["user_id", "item_id"]).sort_index()
        return df
    @timer
    def query_history(self, user_id_list, item_id_list):
        # 1. 构造目标查询表
        target_df = pd.DataFrame(
            {
                "user_id": user_id_list,
                "item_id": item_id_list,
            }
        )
        # 这是底层高度优化的索引匹配，速度极快且不会有 reindex 的类型推导 bug
        query_res = target_df.join(self.df, on=["user_id", "item_id"], how="left")
        history_list = query_res["history"].to_list()
        batch_size = len(history_list)
        res_tensor = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        for i, seq in enumerate(history_list):
            if isinstance(seq, list):
                seq_len = min(len(seq), self.max_len)
                res_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        return res_tensor

iq = InterQuery(inter_path="dataset/ml-1m/ml-1m.inter", max_len=10)
print(iq.df.head(100).to_string())

print(iq.df.info())
user_id_list = [1736, 6039] * 1000
item_id_list = [748, 2804] * 1000
#
import torch
user_id_list = torch.tensor(user_id_list)
item_id_list = torch.tensor(item_id_list)
res = iq.query_history(user_id_list, item_id_list)
print(res)