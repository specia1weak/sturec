## ContextSeqEmbLayer

数据集中所有用户和所有物品的完整特征表（user_feat, item_feat）全部加载并“驻留”在 GPU 显存里。

forward根据
```python
self.embedding_layer = ContextSeqEmbLayer(
    dataset=dataset,             # 传入那个包含元数据的万能 dataset
    embedding_size=64,           # 你想要的统一 Embedding 维度
    pooling_mode='mean',         # 处理多值特征的方式（求均值），例如单字段多标签（科技、电影）
    device=self.device
)

sparse_emb, dense_emb: Dict[{'user', 'item'}, Tensor] = self.embedding_layer(user_idx, item_idx)

# 输入的id -> [feature_nums, D]
user_idx = [B, L]
sparse_emb['item'] = [B, L, num_features, D]
```


ContextSeqEmbLayer是写死的玩意，如果你有自定义的要求，可以参考他的实现代码,它使用了这些Embedding
你可以通过dataset.fields()来获取你需要的特征
FMEmbedding
FLEmbedding