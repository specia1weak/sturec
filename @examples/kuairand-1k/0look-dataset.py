from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset

from betterbole.utils import set_all
set_all()

def build_dataset(device="cpu"):
    override_cfg = {
        "eval_args": {
            "split": {'RS': [0.8, 0.1, 0.1]},
            "order": "TO",
            "mode": "labeled",
        },
        "SCENARIO_FIELD": "tab",
        "train_neg_sample_args": None
    }
    config = Config("BPR", dataset="KuaiRand-1k", config_file_list=["dataset/KuaiRand-1k/KuaiRand-1k.yaml"], config_dict=override_cfg)
    dataset = create_dataset(config)
    if device == "cpu":
        config["device"] = "cpu"
    return config, dataset

if __name__ == '__main__':
    config, dataset = build_dataset()
    dataset: Dataset = dataset

    train_dataset, valid_dataset, test_dataset = data_preparation(config, dataset)

    for batch_data in train_dataset:
        print(batch_data)
        break


