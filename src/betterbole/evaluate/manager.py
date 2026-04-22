class EvaluatorManager:
    def __init__(self):
        self.registry = {}

    def register(self, name, evaluator, filter_fn=None):
        """
        注册评估器。
        filter_fn: 接收 batch_inter，返回布尔掩码
        """
        if filter_fn is None:
            filter_fn = lambda inter: True

        self.registry[name] = {
            "evaluator": evaluator,
            "filter_fn": filter_fn
        }

    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None, batch_inter=None):
        """
        前4个参数完全对齐你的基础 Evaluator。
        唯一多出的 batch_inter 仅作为计算 mask 的上下文，不参与任何解包。
        """
        if batch_inter is None:
            raise ValueError("必须传入 batch_inter 以便 filter_fn 计算掩码。")

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
        final_results = {}
        for name, config in self.registry.items():
            res = config["evaluator"].summary()
            for k, v in res.items():
                final_results[f"[{name}]_{k}"] = v

        if epoch is not None: final_results['epoch'] = epoch
        if step is not None: final_results['step'] = step
        return final_results

    def clear(self):
        for config in self.registry.values():
            config["evaluator"].clear()