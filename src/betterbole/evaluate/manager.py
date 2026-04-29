from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import polars as pl


class EvaluatorFilter(ABC):
    @abstractmethod
    def __call__(self, inter):
        raise NotImplementedError


class AllowAllFilter(EvaluatorFilter):
    def __call__(self, inter):
        return True


class DomainFilter(EvaluatorFilter):
    def __init__(self, field_name: str, domain_id: int):
        self.field_name = field_name
        self.domain_id = domain_id

    def __call__(self, inter):
        return inter[self.field_name] == self.domain_id

class EvaluatorManager:
    def __init__(self, log_path=None, title="untitled"):
        self.registry = {}
        self.log_path = log_path
        self.title = title
        self._is_first_summary = True

        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def register(self, name, evaluator, filter_fn=None):
        """
        注册评估器。
        filter_fn: 接收 batch_inter，返回布尔掩码
        """
        if filter_fn is None:
            filter_fn = AllowAllFilter()

        self.registry[name] = {
            "evaluator": evaluator,
            "filter_fn": filter_fn
        }

    def collect(self, batch_users, batch_targets, batch_inter, batch_preds_1d=None, batch_scores_2d=None, ):
        """
        前4个参数完全对齐你的基础 Evaluator。
        唯一多出的 batch_inter 仅作为计算 mask 的上下文，不参与任何解包。
        """
        for name, config in self.registry.items():
            evaluator = config["evaluator"]
            filter_fn = config["filter_fn"]

            # 1. 仅把 inter 交给 filter_fn 算掩码，绝不从 inter 里取真实数据
            mask = filter_fn(batch_inter)

            # 2. 全量不过滤的情况
            if isinstance(mask, bool) and mask is True:
                evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
                continue

            # 提前拦截空掩码
            if isinstance(mask, bool) and mask is False:
                continue
            if hasattr(mask, 'any') and not mask.any():
                continue

            # 3. 对传入的张量进行原生的布尔切片
            e_users = batch_users[mask]
            e_targets = batch_targets[mask]
            e_preds_1d = batch_preds_1d[mask] if batch_preds_1d is not None else None
            e_scores_2d = batch_scores_2d[mask] if batch_scores_2d is not None else None

            # 4. 完美对齐调用底层的 Evaluator
            evaluator.collect(e_users, e_targets, e_preds_1d, e_scores_2d)

    def summary(self, epoch=None, step=None):
        """
        收集所有指标，追加 epoch/step，并根据配置化的设定自动写文件
        """
        # 1. 收集干净的底层指标
        final_results = {}
        for name, config in self.registry.items():
            res = config["evaluator"].summary()
            for k, v in res.items():
                final_results[f"[{name}]_{k}"] = v

        # 2. 追加进度信息
        if epoch is not None:
            final_results['epoch'] = epoch
        if step is not None:
            final_results['step'] = step

        # 3. 拦截并落盘
        if self.log_path:
            self._write_polars_log(final_results, epoch, step)

        return final_results

    def _write_polars_log(self, metrics_dict, epoch, step):
        """内部方法：专门负责将指标转成 Polars 字符串并追加到文件"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将字典转为纵向结构，方便阅读
        df = pl.DataFrame({
            "Metric": list(metrics_dict.keys()),
            "Value": list(metrics_dict.values())
        })

        # 临时配置 Polars 渲染参数：
        # tbl_rows=100：防止指标太多中间被折叠成 "..."
        # tbl_hide_dataframe_shape=True：隐藏底部的 (shape: 10, 2) 这种无关信息
        with pl.Config(tbl_rows=100, tbl_hide_dataframe_shape=True):
            table_str = str(df)

        # 写入文件
        with open(self.log_path, 'a', encoding='utf-8') as f:
            # 首次记录打印实验 Header
            if self._is_first_summary:
                f.write(f"\n========== 🚀 [{self.title}] 开始记录 | {current_time} ==========\n")
                self._is_first_summary = False

            # 打印当前 Step/Epoch 的信息头
            header_parts = [f"\n[{current_time}]"]
            if epoch is not None: header_parts.append(f"Epoch: {epoch:02d}")
            if step is not None: header_parts.append(f"Step: {step}")

            f.write(" | ".join(header_parts) + "\n")

            # 写入 Polars 格式化好的表格
            f.write(table_str + "\n")

    def clear(self):
        for config in self.registry.values():
            config["evaluator"].clear()
