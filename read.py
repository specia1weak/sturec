from pathlib import Path

import pandas as pd
# 设置显示最大行数为 None (代表无限制)
pd.set_option('display.max_rows', None)
# 设置显示最大列数为 None
pd.set_option('display.max_columns', None)
# 设置列宽，防止内容过长被截断
pd.set_option('display.max_colwidth', None)
# 设置显示宽度，确保不换行
pd.set_option('display.width', 1000)




def parse_feature(feature_list):
    """
    [{'feature_id': 6, 'feature_value_type': 'int_value', 'float_value': None, 'int_array': None, 'int_value': 96.0},
    """
    ret = {}
    def parse_arr(feature_info, key, post_func=None):
        feature_arr = feature_info.get(key)
        if feature_arr is None:
            feature_arr = []

        if post_func is not None:
            return list(map(post_func, feature_arr))
        else:
            return feature_arr


    for feature_info in feature_list:
        feature_name = f"f{feature_info['feature_id']}"
        val_type = feature_info["feature_value_type"]

        feature_value = None
        if val_type == "int_value":
            feature_value = int(feature_info[val_type])
        elif val_type == "float_value":
            feature_value = float(feature_info[val_type])

        if "int_array_and_float_array" in val_type:
            feature_int_arr = parse_arr(feature_info, "int_array", int)
            feature_float_arr = parse_arr(feature_info, "float_array", float)
            feature_value = {"tag": feature_int_arr, "val": feature_float_arr}
        elif "int_array" == val_type:
            feature_value = parse_arr(feature_info, "int_array", int)
        elif "float_array" == val_type:
            feature_value = parse_arr(feature_info, "float_array", float)

        ret[feature_name] = feature_value
    return ret

def parse_label(label_list):
    """
    [{'action_time': 1770694249, 'action_type': 1}]
    """
    return label_list[0]["action_type"]

def parse_seq(seq_list):
    """
    {'action_seq': [], 'content_seq': [], 'item_seq': []}
    :param seq_list:
    :return:
    """
    ret = {}
    for k, feature_list in seq_list.items():
        ret[k] = parse_feature(feature_list)
    return ret

if __name__ == "__main__":
    df = pd.read_parquet("raw/kdd26cup/sample_data.parquet") # 全1序列

    # processed_df = pd.DataFrame(columns=['item_id', 'item_feature', 'label', 'seq_feature', 'timestamp', 'user_feature', 'user_id'])

    print(df.shape)  # (1000, 7)
    print(df.columns)
    user_id = df["user_id"].value_counts()
    item_id = df["item_id"].value_counts()
    df["label"] = df["label"].apply(parse_label)
    df["item_feature"] = df["item_feature"].apply(parse_feature)
    df["user_feature"] = df["user_feature"].apply(parse_feature)
    df["seq_feature"] = df["seq_feature"].apply(parse_seq)
    label = df["label"].value_counts()

    # print(user_id)
    # print(item_id)
    print(label)
    # print(df.head(1)["item_feature"])
    # print(df.head(1)["user_feature"])

    print(df.head(1)["seq_feature"])