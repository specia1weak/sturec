from pathlib import Path
ROOT_DIR = Path(__file__).parents[3]
WORKSPACE = ROOT_DIR / "workspace"

def ignore_future_warning():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

def change_root_workdir():
    import os
    print("切换到: ", ROOT_DIR)
    os.chdir(ROOT_DIR)

def set_all():
    ignore_future_warning()
    change_root_workdir()

def preset_workdir(dataset_name: str="untitled"):
    return WORKSPACE / dataset_name

if __name__ == '__main__':
    print(preset_workdir())