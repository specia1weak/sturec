import torch
import torch.nn as nn
from recbole.data.dataset import Dataset
from recbole.trainer import Trainer
from torch import optim
from torch.nn.init import xavier_normal_, constant_
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers, SequenceAttLayer, ContextSeqEmbAbstractLayer
from recbole.utils import InputType, FeatureType

class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset: Dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            "user": list(self.user_feat.interaction.keys()),
            "item": list(self.item_feat.interaction.keys()),
        }

        self.types = ["user", "item"]

        print(self.dataset.fields())
        if "timestamp" in self.dataset.fields():
            self.dataset.config["numerical_features"] = ["timestamp"]
            self.field_names.update({
                "inter": ["timestamp"]
            })
            self.types.append("inter")

        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()

class DIN(SequentialRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(DIN, self).__init__(config, dataset)

        # get field names and parameter value from config
        self.LABEL_FIELD = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.device = config["device"]
        self.dropout_prob = config["dropout_prob"]

        self.types = ["user", "item"]
        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        # init MLP layers
        # self.dnn_list = [(3 * self.num_feature_field['item'] + self.num_feature_field['user'])
        #                  * self.embedding_size] + self.mlp_hidden_size
        num_item_feature = sum(
            (
                1
                if dataset.field2type[field]
                not in [FeatureType.FLOAT_SEQ, FeatureType.FLOAT]
                or field in config["numerical_features"]
                else 0
            )
            for field in self.item_feat.interaction.keys()
        )
        self.dnn_list = [
            3 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size
        self.att_list = [
            4 * num_item_feature * self.embedding_size
        ] + self.mlp_hidden_size

        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask
        self.attention = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=False,
            return_seq_weight=False,
        )
        self.dnn_mlp_layers = MLPLayers(
            self.dnn_list, activation="Dice", dropout=self.dropout_prob, bn=True
        )

        self.embedding_layer = ContextSeqEmbLayer( # 加载Embedding字典和查表器，好像只对最后一维的idx进行查表
            dataset, self.embedding_size, "mean", self.device
        )
        self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ["embedding_layer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, item_seq_len, next_items):
        max_length = item_seq.shape[1]
        # concatenate the history item seq with the target item to get embedding together
        item_seq_next_item = torch.cat((item_seq, next_items.unsqueeze(1)), dim=-1)
        sparse_embedding, dense_embedding = self.embedding_layer(
            user, item_seq_next_item
        )
        # concat the sparse embedding and float embedding
        feature_table = {}
        for type in self.types:
            feature_table[type] = []
            if sparse_embedding[type] is not None:
                feature_table[type].append(sparse_embedding[type])
            if dense_embedding[type] is not None:
                feature_table[type].append(dense_embedding[type])

            feature_table[type] = torch.cat(feature_table[type], dim=-2)
            table_shape = feature_table[type].shape
            feat_num, embedding_size = table_shape[-2], table_shape[-1]
            feature_table[type] = feature_table[type].view(
                table_shape[:-2] + (feat_num * embedding_size,)
            )

        user_feat_list = feature_table["user"]
        item_feat_list, target_item_feat_emb = feature_table["item"].split(
            [max_length, 1], dim=1
        )
        target_item_feat_emb = target_item_feat_emb.squeeze(1)

        # attention
        user_emb = self.attention(target_item_feat_emb, item_feat_list, item_seq_len)
        user_emb = user_emb.squeeze(1)

        # input the DNN to get the prediction score
        din_in = torch.cat(
            [user_emb, target_item_feat_emb, user_emb * target_item_feat_emb], dim=-1
        )
        din_out = self.dnn_mlp_layers(din_in)
        preds = self.dnn_predict_layers(din_out)

        return preds.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL_FIELD]
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        output = self.forward(user, item_seq, item_seq_len, next_items)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        next_items = interaction[self.POS_ITEM_ID]
        scores = self.sigmoid(self.forward(user, item_seq, item_seq_len, next_items))
        return scores


train_cfg = {
    "train_batch_size": 2048,
    # "gpu_id": "",  # CPU 运行
    "embedding_size": 64,
    "MAX_ITEM_LIST_LENGTH": 10,  # 核心：强制开启并限制序列最大长度
    # 序列推荐通常使用 leave-one-out (留一法) 划分数据集，这里确保显式覆盖配置
    "eval_args": {
        "split": {"LS": "valid_and_test"},
        "order": "TO",
        "mode": "labeled"
    },
    "val_interval": {
        "rating": '[4.0, 5.0]'
    },
    "train_neg_sample_args": None,  # 明确告诉框架：CE 损失不需要训练负采样,

}

if __name__ == '__main__':
    from src.utils import change_root_workdir, ignore_future_warning

    change_root_workdir()
    ignore_future_warning()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation

    # 【关键 3】直接把模型类 MySimpleSeqRecommender 传给 Config，而不是传字符串
    config = Config(model=DIN, dataset='ml-1m', config_file_list=['dataset/ml-1m/m1-1m.yaml'],
                    config_dict=train_cfg)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = DIN(config, dataset).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(config, model)

    print(f"训练集大小: {len(train_data.dataset)}, 序列最大长度: {config['MAX_ITEM_LIST_LENGTH']}")

    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for batch_idx, batch_data in enumerate(train_data):
            # 将数据推到正确的设备 (CPU/GPU)
            batch_data = batch_data.to(config['device'])
            loss = model.calculate_loss(batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)

        # 验证阶段
        result = trainer.evaluate(
            valid_data,
            load_best_model=False,
            show_progress=False
        )
        print(f"Epoch {epoch + 1:02d} | 训练 Loss: {avg_train_loss:.4f} | 验证集指标: {result}")
        print("-" * 50)
