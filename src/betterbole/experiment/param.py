import argparse
from dataclasses import dataclass, fields


@dataclass
class ExperimentConfig:
    experiment_name: str = "untitled"
    seed: int = 2026


class ParamManager:
    def __init__(self, config_class):
        self.config_class = config_class

    def build(self, **code_kwargs):
        # 提取数据类中定义的所有字段和默认值
        config_dict = {f.name: f.default for f in fields(self.config_class)}
        valid_code_kwargs = {k: v for k, v in code_kwargs.items() if k in config_dict}
        config_dict.update(valid_code_kwargs)

        # 3. 第三层：CLI 命令行传参覆盖
        parser = argparse.ArgumentParser()
        for f in fields(self.config_class):
            # 处理布尔类型的特殊情况
            if f.type is bool:
                parser.add_argument(
                    f"--{f.name}",
                    type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                    default=argparse.SUPPRESS
                )
            else:
                parser.add_argument(f"--{f.name}", type=f.type, default=argparse.SUPPRESS)

        # 解析命令行参数 (parse_known_args 允许传入未定义的参数而不报错)
        cli_args, _ = parser.parse_known_args()
        config_dict.update(vars(cli_args))
        return self.config_class(**config_dict)

    def __str__(self):
        lines = [f"ParamManager managing <{self.config_class.__name__}>"]
        lines.append("-" * 50)
        lines.append(f"{'Parameter':<18} | {'Type':<8} | {'Default'}")
        lines.append("-" * 50)
        for f in fields(self.config_class):
            type_name = getattr(f.type, '__name__', str(f.type))
            import dataclasses
            if f.default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
                default_val = "<factory>"
            elif f.default is dataclasses.MISSING:
                default_val = "<required>"
            else:
                default_val = repr(f.default)
            lines.append(f"{f.name:<18} | {type_name:<8} | {default_val}")
        lines.append("-" * 50)
        return "\n".join(lines)


if __name__ == '__main__':
    manager = ParamManager(ExperimentConfig)
    manager.build()
    print(manager)
