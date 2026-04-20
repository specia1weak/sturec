# ==========================================
# 2. 调度层 (Manager Layer) - 计算图统筹
# ==========================================
import json
from pathlib import Path
from typing import Union, Literal

import polars as pl

from betterbole.data.split import SPLIT_STRATEGIES, SplitContext
from betterbole.emb.schema import IdSeqEmbSetting, SparseEmbSetting, SparseSetEmbSetting, EmbSetting, EmbType
from betterbole.core.enum_type import FeatureSource
from typing import Any, List, Iterable
def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]

class SchemaManager:
    ITEM_PROFILE_NAME = "item_profile.parquet"
    USER_PROFILE_NAME = "user_profile.parquet"
    SCHEMA_META_NAME = "feature_meta.json"
    WHOLE_DATA_NAME = "whole_dataframe.parquet"
    def __init__(self, settings_list: List[EmbSetting], work_dir: str,
                 time_field=None, label_fields: Union[str, Iterable[str]]=None,
                 domain_fields: Union[str, Iterable[str]]=None):
        self.settings = settings_list

        ## path & dir
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.meta_filepath = self.work_dir / self.SCHEMA_META_NAME

        ## field_name
        self.uid_field = next(
            (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
            None
        )
        self.iid_field =next(
            (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
            None
        )
        self.time_field = time_field
        self._label_fields = ensure_list(label_fields)
        self._domain_fields = ensure_list(domain_fields)

    @property
    def label_field(self):
        return next(iter(self._label_fields), None)
    @property
    def label_fields(self) -> tuple:
        return tuple(self._label_fields)
    @property
    def domain_field(self):
        return next(iter(self._domain_fields), None)
    @property
    def domain_fields(self) -> tuple:
        return tuple(self._domain_fields)


    def _parse_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.work_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir


    def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
        """
        不建议使用，全自动化流程：自动判断输入文件类型，自动构建计算图
        """
        output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
        if self.meta_filepath.exists() and output_parquet.exists() and not redo:
            print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
            self.load_schema()
            return pl.scan_parquet(output_parquet)

        print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
        lazy_df = self._make_checkpoint(lazy_df)

        print("[*] 阶段一：构建 Fit 计算图...")
        fit_exprs = []
        for setting in self.settings:
            if not setting.is_fitted:
                fit_exprs.extend(setting.get_fit_exprs())

        if fit_exprs:
            print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
            fit_result = lazy_df.select(fit_exprs).collect()
            for setting in self.settings:
                if not setting.is_fitted:
                    setting.parse_fit_result(fit_result)
            print("[+] 统计量计算与映射表构建完成。")
        else:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
        print("[*] 阶段二：构建 Transform 计算图...")
        transform_exprs = [
            setting.get_transform_expr()
            for setting in self.settings
            if setting.emb_type != EmbType.UNKNOWN
        ]
        print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
        lazy_df.with_columns(transform_exprs) \
            .sink_parquet(output_parquet)
        self.save_schema()
        print("[+] 预处理全流程结束！")
        return pl.scan_parquet(output_parquet)

    def fit(self, train_raw_lf: pl.LazyFrame):
        """
        阶段一：仅使用训练集拟合统计量和词表
        """
        if self.meta_filepath.exists():
            self.load_schema()
            return
        print("[*] 正在执行计算图截断，避免重复计算...")
        print("[*] 启动单次表扫描获取统计量...")
        fit_exprs = [expr for s in self.settings if not s.is_fitted for expr in s.get_fit_exprs()]
        if fit_exprs:
            fit_result = train_raw_lf.select(fit_exprs).collect()
            for setting in self.settings:
                if not setting.is_fitted:
                    setting.parse_fit_result(fit_result)
            print("[+] 词表与映射规则构建完成。")
        self.save_schema()

    def transform(self, raw_lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        阶段二：根据已固化的 Schema 进行流式转换
        """
        transform_exprs = [
            s.get_transform_expr() for s in self.settings if s.emb_type != EmbType.UNKNOWN
        ]
        return raw_lf.with_columns(transform_exprs)

    def _make_checkpoint(self, lazy_df, file_name="temp_checkpoint.parquet"):
        temp_checkpoint_path = self.work_dir / file_name
        lazy_df.sink_parquet(temp_checkpoint_path)
        lazy_df = pl.scan_parquet(temp_checkpoint_path)
        return lazy_df

    # NOTE 不推荐使用
    def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
        """
        利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
        """
        item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
        user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME

        print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
        # 1. Item Profile
        item_features = [setting.field_name for setting in self.settings
                         if setting.source == FeatureSource.ITEM]
        if self.iid_field and (not item_profile_path.exists() or redo):
            # 确保主键在提取列表中
            if self.iid_field not in item_features:
                item_features.insert(0, self.iid_field)

            print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
            (
                lazy_df
                .select(item_features)
                .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
                .unique(subset=[self.iid_field])  # 按物品 ID 去重
                .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
                .sink_parquet(item_profile_path)
            )
        # 2. 提取 User Profile (同理)
        user_features = [
            setting.field_name
            for setting in self.settings
            if setting.source == FeatureSource.USER
        ]

        if self.uid_field and (not user_profile_path.exists() or redo):
            if self.uid_field not in user_features:
                user_features.insert(0, self.uid_field)

            print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
            (
                lazy_df
                .select(user_features)
                .drop_nulls(subset=[self.uid_field])
                .unique(subset=[self.uid_field])
                .sort(self.uid_field)
                .sink_parquet(user_profile_path)
            )
        print("[+] 静态查找表提取完成！")

    def split_dataset(self, lf: pl.LazyFrame, strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
                      output_dir: str = None, redo: bool = False, **kwargs):
        output_dir = self._parse_output_dir(output_dir)
        StrategyClass = SPLIT_STRATEGIES.get(strategy)
        if not StrategyClass:
            raise ValueError(f"未知的切分策略: {strategy}")
        split_ctx = SplitContext(
            uid_field=self.uid_field,
            iid_field=self.iid_field,
            time_field=self.time_field,
            checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
        )
        # 策略实例化与执行
        splitter = StrategyClass(context=split_ctx)
        kwargs.update({
            "time_field": self.time_field or kwargs.get("time_field")
        })
        return splitter.split(lf, output_dir, redo, **kwargs)

    def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
        """
        如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
        """
        output_dir = self._parse_output_dir(output_dir)
        train_path = output_dir / "train.parquet"
        valid_path = output_dir / "valid.parquet"
        test_path = output_dir / "test.parquet"
        split_paths = str(train_path), str(valid_path), str(test_path)
        if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
            return split_paths

        all_lf = [train_lf, valid_lf, test_lf]
        for lf, save_path in zip(all_lf, split_paths):
            if lf is not None:
                lf.sink_parquet(save_path)

        return split_paths

    def save_schema(self):
        meta_data = [s.to_dict() for s in self.settings]
        with open(self.meta_filepath, 'w') as f:
            json.dump(meta_data, f, indent=4)

    def load_schema(self):
        if not self.meta_filepath.exists():
            print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
            return

        with open(self.meta_filepath, 'r', encoding='utf-8') as f:
            meta_data_list = json.load(f)

        meta_dict = {item["field_name"]: item for item in meta_data_list}
        for setting in self.settings:
            if setting.field_name in meta_dict:
                saved_state = meta_dict[setting.field_name]
                if saved_state.get("type") != setting.emb_type.name:
                    raise ValueError(
                        f"特征 {setting.field_name} 的类型不匹配: "
                        f"期望 {setting.emb_type.name}, JSON 中为 {saved_state.get('type')}"
                    )

                setting.load_state(saved_state)
                print(f"[-] 成功恢复特征状态: {setting.field_name}")
            else:
                print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")

    def source2emb_size(self, *sources: FeatureSource):
        emb_size = 0
        for setting in self.settings:
            if setting.source in sources:
                emb_size += setting.embedding_size
        return emb_size

    def get_setting(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self):
        fields = []
        for setting in self.settings:
            fields.append(setting.field_name)
            if isinstance(setting, IdSeqEmbSetting):
                fields.append(setting.seq_len_field_name)
        for ctx_field in (self.time_field, *self.label_fields, *self.domain_fields):
            if ctx_field is not None and ctx_field not in fields:
                fields.append(ctx_field)
        return fields

# ==========================================
# 3. 使用示例 (假设使用 KuaiRand)
# ==========================================
if __name__ == "__main__":
    # 假设特征注册
    user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)

    # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
    tags = SparseSetEmbSetting(
        field_name="tag",
        source=FeatureSource.ITEM,
        is_string_format=True,
        separator=","
    )

    manager = SchemaManager([user_id_setting, tags])

    from betterbole.datasets.kuairand import KuaiRand
    item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
    user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)

    whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
    whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
    # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
    # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
    manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
    print("架构编译成功，可供调用。")

    import pandas as pd
    df = pd.read_parquet("kuairand_processed.parquet")
    print(df.head(3).to_string())