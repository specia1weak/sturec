## 顶层抽象
```python
class AbstractRecommender(nn.Module):
    def __init__(self):
        super(AbstractRecommender, self).__init__()
    def calculate_loss(self, interaction):
        r"""
        返回训练Loss, 0维张量，例如tensor(0.5)
        """

    def predict(self, interaction):
        r"""
        返回打分 [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r""" 
        输入只有用户侧信息的 interaction, 你需要返回 [n_users * n_items]的打分矩阵
        """
        raise NotImplementedError
```
full_sort_predict对一个用户以及数据的所有物品全量打分，但在多场景多域情况下不可用。这个方法只能在双塔（算相似度）里面使用
如果是精排（输出CTR）那么这个方案不再适用。因为双塔召回预测一对UI只需要做点积，而CTR预测一对UI需要走一次模型。
对于CTR，使用AUC和LogLoss是更一般的选择。

recbole框架不会使用你的forward

