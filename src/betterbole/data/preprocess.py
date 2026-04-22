from abc import ABC
from typing import List

from betterbole.emb import SchemaManager
from betterbole.emb.schema import EmbSetting, SparseEmbSetting, SparseSetEmbSetting, IdSeqEmbSetting
from betterbole.core.enum_type import FeatureSource
import polars as pl

from betterbole.utils import extract_history_items


class PreProcessPipeline(ABC):
    def load_and_process_lf(self)->pl.LazyFrame:
        from betterbole.convert.kuairand import KuaiRand
        user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)
        item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
        inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
        whole_lf: pl.LazyFrame = inter_lf.join(item_lf, on="video_id", how="left").join(user_lf, on="user_id",
                                                                                        how="left")
        ## 排除部分场景数据
        top_5_domains = (
            whole_lf.group_by("tab")
            .len()
            .sort("len", descending=True)
            .head(5)
            .select("tab")
        )
        whole_lf = whole_lf.join(top_5_domains, on="tab", how="semi")

        ## 增加新特征
        parsed_date = pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        whole_lf = whole_lf.with_columns(
            parsed_date.dt.weekday().alias("day_of_week"),
            parsed_date.dt.weekday().is_in([6, 7]).cast(pl.UInt8).alias("is_weekend"),
            (pl.col("hourmin").cast(pl.Int32) // 100).cast(pl.UInt8).alias("hour")
        )
        ## 增加新序列特征
        max_seq_len = 50
        whole_lf = extract_history_items(whole_lf, max_seq_len=50,
                                            user_col="user_id",
                                            time_col="time_ms",
                                            item_col="video_id",
                                            seq_col="history_videos")
        return whole_lf

    def set_emb_settings(self)->List[EmbSetting]:
        user_setting = SparseEmbSetting("user_id", FeatureSource.USER_ID, 64)
        item_setting = SparseEmbSetting("video_id", FeatureSource.ITEM_ID, 64)
        settings_list = [
            user_setting,
            item_setting,
            SparseEmbSetting("tab", FeatureSource.INTERACTION, 16),
            SparseEmbSetting("user_active_degree", FeatureSource.USER, 16),
            SparseEmbSetting("is_live_streamer", FeatureSource.USER, 16),
            SparseEmbSetting("is_video_author", FeatureSource.USER, 16),
            SparseEmbSetting("friend_user_num_range", FeatureSource.USER, 16),
            SparseEmbSetting("register_days_range", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat0", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat1", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat2", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat3", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat4", FeatureSource.USER, 16),
            SparseEmbSetting("onehot_feat5", FeatureSource.USER, 16),
            #
            SparseEmbSetting("video_type", FeatureSource.ITEM, 16),
            SparseEmbSetting("upload_type", FeatureSource.ITEM, 16),

            SparseSetEmbSetting(
                "tag", FeatureSource.ITEM, 16,
                is_string_format=True, separator=","
            ),

            # 创造的
            IdSeqEmbSetting("history_videos", FeatureSource.INTERACTION, item_setting,
                            is_string_format=False),
            SparseEmbSetting("day_of_week", FeatureSource.INTERACTION, 16),
            SparseEmbSetting("hour", FeatureSource.INTERACTION, 16),
            SparseEmbSetting("is_weekend", FeatureSource.INTERACTION, 16),
        ]
        return settings_list


    def save_schema(self, settings_list: List[EmbSetting], whole_lf: pl.LazyFrame)->SchemaManager:
        manager = SchemaManager(settings_list, "kuairand-workdir", time_field="time_ms", label_field="is_click",
                                domain_field="tab")
        transformed_lf = manager.prepare_data(whole_lf)
        manager.generate_profiles(transformed_lf)
        return manager