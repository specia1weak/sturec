## Series
```
def __init__(
    self,
    data=None, # 兼容性恐怖的参数，任何东西都能被理解，包括基础的单值、可迭代对象，甚至字典额外会赋予index
    index=None, # 默认从0开始的整数索引，传入长度一致的可哈希key值就行
    dtype: Dtype | None = None, # 默认pandas自行推断
    name=None, # 为这个Series取名，未来变成 df 才会起作用
    copy: bool | None = None, # 是否copy分离
    fastpath: bool | lib.NoDefault = lib.no_default, # 请不要用他
)
```

1. 实际上Series看上去更像一个**可重**有序字典结构，key即index，val即data。
2. name用来整体表示这个Series，仅仅变成Dataframe的时候他才起作用。
3. Series在Dataframe可能以row的形式存在，也可能以column的形式存在，name和index的作用也会倒转
4. 直接索引和.loc机制是替换机制，[["index"]]来获取维持原形状的结果
5. 内存（物理）索引iloc, 相当于对有序字典的value值当做数组, 例如se.iloc[0]

### 可用函数
1. \**.value_counts()  统计完全相同的row的出现频次，因为统计col没有任何意义
2. \**.drop_duplicates()  去除重复row
3. \**.apply(func)  func(row)->row  应用在所有value上，最终返回一个新对象。依靠python循环，所以比较慢
4. \*.map(dict)  将value根据传入的dict映射成另一个ser上的值