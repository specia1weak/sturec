import sys
sys.path.append("..")
import torch
import pandas as pd
from tqdm import tqdm
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scenario_wise_rec.models.multi_domain.adaptdhm import AdaptDHM
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import Star, MMOE, PLE, SharedBottom, AdaSparse, Sarnet, M2M, EPNet, PPNet, M3oE, HamurSmall



def get_kuairand_data_multidomain(data_path="./data/kuairand/"):
    data = pd.read_csv(data_path+"/kuairand_sample.csv")
    data = data[data["tab"].apply(lambda x: x in [1, 0, 4, 2, 6])]
    data.reset_index(drop=True, inplace=True)

    data.rename(columns={'tab': "domain_indicator"}, inplace=True)
    domain_num = data.domain_indicator.nunique()

    col_names = data.columns.to_list()

    dense_features = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]

    useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
    scenario_features = ["domain_indicator"]

    sparse_features = [col for col in col_names if col not in dense_features and
                       col not in useless_features and col not in ['is_click','domain_indicator']]
    # target = "is_click"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in scenario_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]

    y=data["is_click"]
    del data["is_click"]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

def get_kuairand_data_multidomain_ppnet(data_path="./data/kuairand/"):
    data = pd.read_csv(data_path+"/kuairand_sample.csv")
    data = data[data["tab"].apply(lambda x: x in [1, 0, 4, 2, 6])]
    data.reset_index(drop=True, inplace=True)

    data.rename(columns={'tab': "domain_indicator"}, inplace=True)
    domain_num = data.domain_indicator.nunique()

    col_names = data.columns.to_list()

    dense_features = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]

    useless_features = ["play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time"]
    scenario_features = ["domain_indicator"]
    id_features = ["user_id", "video_id"]

    sparse_features = [col for col in col_names if col not in dense_features and
                       col not in useless_features and col not in id_features and
                       col not in ['is_click']]
    # target = "is_click"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in scenario_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
    for feature in id_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    id_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in id_features]
    y=data["is_click"]
    del data["is_click"]

    return dense_feas, sparse_feas, scenario_feas, id_feas, data, y, domain_num