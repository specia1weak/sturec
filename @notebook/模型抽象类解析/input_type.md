```python
class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3
```
PAIRWISE需要设计neg_iid的采样方式，负采样会同时把负样本item的其他特征放进interaction中。并默认以neg为前缀
- GeneralRecommender是PAIRWISE的，General只关心排序问题，给一个正样本负样本，模型能区分就行
- InputType实际上影响的dataloader的逻辑, 任何一个继承自模型都应当有一个类属性input_type。当然你也可以不指定，但是一定不能将没有input_type的cls传给config。

