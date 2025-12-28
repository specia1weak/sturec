import pickle
from pathlib import Path

# 路径需与你实际一致
BASE_DIR = Path("D:/pyprojects/recommend-study/studybole/dataset/Amazon_Processed_10Core/Pretrained_Embeddings")
SOURCE_PATH = BASE_DIR / "Source_Books_user_emb.pkl"
TARGET_PATH = BASE_DIR / "Target_Movies_user_emb.pkl"


def check():
    print(f"Loading {SOURCE_PATH}...")
    with open(SOURCE_PATH, 'rb') as f:
        s_dict = pickle.load(f)

    print(f"Loading {TARGET_PATH}...")
    with open(TARGET_PATH, 'rb') as f:
        t_dict = pickle.load(f)

    s_keys = list(s_dict.keys())
    t_keys = list(t_dict.keys())

    print("\n[--- 诊断信息 ---]")
    print(f"Source 字典大小: {len(s_keys)}")
    print(f"Target 字典大小: {len(t_keys)}")

    print(f"\nSource Key 示例 (前3个): {s_keys[:3]}")
    print(f"Source Key 类型: {type(s_keys[0])}")

    print(f"\nTarget Key 示例 (前3个): {t_keys[:3]}")
    print(f"Target Key 类型: {type(t_keys[0])}")

    # 尝试强制转换后的交集
    s_set_str = set(str(k) for k in s_keys)
    t_set_str = set(str(k) for k in t_keys)
    overlap = s_set_str & t_set_str
    print(f"\n强制转换为 str 后的重叠数量: {len(overlap)}")


if __name__ == "__main__":
    check()