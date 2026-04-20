def ignore_future_warning():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

def change_root_workdir():
    import os
    from pathlib import Path
    print("切换到: ", Path(__file__).parents[3])
    os.chdir(Path(__file__).parents[3])

def set_all():
    ignore_future_warning()
    change_root_workdir()