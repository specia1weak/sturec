import argparse
import dataclasses
from dataclasses import dataclass, fields, field, asdict
from typing import get_args, get_origin, Literal, get_type_hints


@dataclass
class ConfigBase:
    experiment_name: str = "untitled" # 提示这次实验的名字
    dataset_name: str = "unknown" # 影响workdir的设置
    seed: int = 2026
    device: str = "cpu"
    max_epochs: int = 100

    extras: dict = field(default_factory=dict)
    def __getattr__(self, name):
        if name in self.extras:
            return self.extras[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        config_dict = asdict(self)
        lines = [f"\n🚀 [Active Config State]: <{self.__class__.__name__}>"]
        lines.append("-" * 50)
        max_key_len = max([len(str(k)) for k in config_dict.keys()] + [10])

        for key, value in config_dict.items():
            if isinstance(value, type):
                display_val = f"<class '{value.__name__}'>"
            elif callable(value):
                display_val = f"<callable '{getattr(value, '__name__', str(value))}'>"
            elif isinstance(value, dict) and key == "extras":
                if not value:
                    display_val = "{}"
                else:
                    display_val = "{" + ", ".join(f"'{k}': {repr(v)}" for k, v in value.items()) + "}"
            else:
                display_val = repr(value)  # 使用 repr 可以让字符串带上引号，更严谨
            lines.append(f"  {key:<{max_key_len}} = {display_val}")
        lines.append("-" * 50)
        return "\n".join(lines)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        annotations = getattr(cls, '__annotations__', {})
        missing_annotations = []
        for key, value in cls.__dict__.items():
            if key.startswith('_') or callable(value) or isinstance(value, (property, classmethod, staticmethod)):
                continue
            if key not in annotations:
                missing_annotations.append(key)
        if missing_annotations:
            missing_str = ", ".join(f"'{f}'" for f in missing_annotations)
            raise TypeError(
                f"🚨 拦截: 数据类 '{cls.__name__}' 存在未加类型注解的属性！\n"
                f"缺失注解的字段共 {len(missing_annotations)} 个: [{missing_str}]\n"
                f"👉 修复建议: 请修改为 '字段名: 类型 = ...'，如无特定类型请使用 'Any'。"
            )

def seed_everything(seed=42):
    import os
    import numpy as np
    import random
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""
调用build方法可以从任何地方获取参数
"""


class ParamManager:
    def __init__(self, config_class):
        self.config_class = config_class
        self.resolved_type_hints = get_type_hints(config_class)

        # 1. 基础类型转换注册表：消除对 bool 等基本类型的 if-else
        self.type_parsers = {
            bool: lambda x: str(x).lower() in ('true', '1', 'yes', 'y', 't'),
            list: lambda x: x.split(','),
        }

        # 2. 实例工厂注册表：{ 字段名: { 字符串key: 实例化方法/类 } }
        self.instance_registry = {}

    def register(self, field_name: str, mapping_dict: dict):
        """
        优雅的字符串到实例注册方法
        :param field_name: dataclass 中的字段名
        :param mapping_dict: 字符串参数到类/工厂函数的映射字典
        """
        self.instance_registry[field_name] = mapping_dict
        return self

    def _get_field_type(self, field):
        return self.resolved_type_hints.get(field.name, field.type)

    def _get_parser_action(self, field):
        """决定如何将命令行的字符串解析为目标对象"""
        field_type = self._get_field_type(field)

        # 优先级1：如果字段被注册了实例映射器 (例如 optimizer 字段)
        if field.name in self.instance_registry:
            mapping = self.instance_registry[field.name]
            def instance_factory(x):
                if x not in mapping:
                    raise ValueError(f"Invalid option '{x}' for {field.name}. Available: {list(mapping.keys())}")
                return mapping[x]
            return instance_factory
        if get_origin(field_type) is Literal:
            literal_values = get_args(field_type)
            if not literal_values:
                raise ValueError(f"Literal field '{field.name}' has no candidate values")
            return type(literal_values[0])
        if get_origin(field_type) is list:
            return self.type_parsers[list]
        if field_type in self.type_parsers:
            return self.type_parsers[field_type]
        return field_type

    def build(self, **code_kwargs):
        # 1. 【修复点】安全提取基类默认字段（同时兼容 default 和 default_factory）
        config_dict = {}
        for f in fields(self.config_class):
            if f.default is not dataclasses.MISSING:
                config_dict[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                config_dict[f.name] = f.default_factory()  # 这里会安全地生成 {}

        converters = {f.name: self._get_parser_action(f) for f in fields(self.config_class)}

        extras_dict = {}

        # 2. 处理代码直接传入的参数 (code_kwargs)
        for k, v in code_kwargs.items():
            if k in self.instance_registry and isinstance(v, str):
                mapping = self.instance_registry[k]
                if v not in mapping:
                    raise ValueError(f"Invalid option '{v}' for '{k}'. Available: {list(mapping.keys())}")
                processed_v = mapping[v]
            else:
                processed_v = v

            # 分发结果
            if k in config_dict:
                config_dict[k] = processed_v
            elif k != "extras":  # 防止覆盖本身的 extras 字段
                extras_dict[k] = processed_v

        # 3. 处理命令行参数
        parser = argparse.ArgumentParser()
        for f in fields(self.config_class):
            parser.add_argument(f"--{f.name}", type=converters[f.name], default=argparse.SUPPRESS)

        known_field_names = {f.name for f in fields(self.config_class)}
        for reg_k, mapping in self.instance_registry.items():
            if reg_k not in known_field_names:
                def dynamic_factory(x, m=mapping, key=reg_k):
                    if x not in m: raise ValueError(f"Invalid '{key}': {x}")
                    return m[x]

                parser.add_argument(f"--{reg_k}", type=dynamic_factory, default=argparse.SUPPRESS)

        cli_args, _ = parser.parse_known_args()

        # 4. 合并命令行参数解析结果
        for k, v in vars(cli_args).items():
            if k in config_dict:
                config_dict[k] = v
            elif k != "extras":
                extras_dict[k] = v

        # 5. 【现在的 config_dict["extras"] 绝对是一个字典了】
        if "extras" in config_dict:
            config_dict["extras"].update(extras_dict)

        # 6. 实例化
        cfg = self.config_class(**config_dict)
        if hasattr(cfg, "seed"):
            seed_everything(cfg.seed)
        return cfg
    def __str__(self):
        lines = [f"ParamManager managing <{self.config_class.__name__}>"]
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<18} | {'Type':<12} | {'Default'}")
        lines.append("-" * 60)
        for f in fields(self.config_class):
            field_type = self._get_field_type(f)
            type_name = getattr(field_type, '__name__', str(field_type))
            if f.default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
                default_val = "<factory>"
            elif f.default is dataclasses.MISSING:
                default_val = "<required>"
            else:
                default_val = repr(f.default)
            lines.append(f"{f.name:<18} | {type_name:<12} | {default_val}")
        lines.append("-" * 60)
        return "\n".join(lines)

if __name__ == '__main__':
    from betterbole.models.backbone import SharedBottomLess
    manager = ParamManager(ConfigBase) # 必须是出现在ConfigBase通过doc声明的方式才能被探测到
    manager.register("model", {
        "sharedbottom": SharedBottomLess
    })
    cfg = manager.build(model="sharedbottom")
    print(manager)
    print(cfg)
