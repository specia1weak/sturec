```python
class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]
```

- General的定义比较简陋，他只是简单包装了AbstractRecommender的初始化。
- 他只关心ID，在初始化上就可以看出来。
- 他额外多了self.NEG_ITEM_ID，这不意味他只能使用PAIRWISE，而是说你要用PAIRWISE的话默认提供了这个变量