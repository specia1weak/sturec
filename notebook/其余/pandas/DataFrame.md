## Dataframe
```python
def __init__(
    self,
    data=None,  # data兼容性高，不过至少传入1维数据，就像Series允许传入0维值。允许传入字典额外赋予columns
    index: Axes | None = None, # Iterable[Hashable]即可
    columns: Axes | None = None, # Iterable[Hashable]即可，额外能filter掉字典data的key值
    dtype: Dtype | None = None,
    copy: bool | None = None,
)
```

1. DataFrame是二维机制，你依然可以视为可重有序字典，不过value全是Series。


### 可用函数
1. \**.value_counts()  统计value的出现频次 -> 类似Counter(ser)
2. \**.drop_duplicates()  去除重复value
3. \**.apply(func, axic=1)  func(row)->v/ser  如果你返回非ser（即便你返回其他Iterable），apply最终返回一个ser；如果你返回ser，最终返回DataFrame
4. \**.apply(func, axic=0)  func(col)  一般不常用
5. .drop(columns=[] or index=[]) 
6. .reset_index(drop=True) 抽离出当前的index，追加新的从0-n的index，如果drop则扔掉被抽离的index

#### Part1 数据探查
1. df.head(n), df.tail(n)  查看前后几行
2. df.describe()->DF 统计均值、标准差、最小值、最大值、25%/50%/75% 分位数）
3. df.shape / df.dtypes / df.columns 查看基础属性
4. df.info() 对一个df的简短概括
5. 查看完整的输出
```python
# 设置显示最大行数为 None (代表无限制)
pd.set_option('display.max_rows', None)
# 设置显示最大列数为 None
pd.set_option('display.max_columns', None)
# 设置列宽，防止内容过长被截断
pd.set_option('display.max_colwidth', None)
# 设置显示宽度，确保不换行
pd.set_option('display.width', 1000)

或者df.to_string()
```

#### Part2 缺失值处理
1. df.isna()/df.isnull()->DF[Bool] 返回形状相同的Bool DF
2. df.dropna(axis=0, subset=None, thresh=None), 在subset的rows中删除缺失值个数 >= thresh; 注意axis=0是按行往下走和apply的axis=1有区别
3. df.fillna(value=None, method=None)
4. df.interpolate()

#### Part3 查询
1. df.query('age > 10 and age < @out_var'), 内部变量代表ser，使用@out_var会找上下文变量. 只查行


#### Part4 分组与聚合
##### 1. df.groupby(key)  迭代器对象，使用for循环动态给出结果
```python
dfg = df.groupby("吉吉国王")
for group_key, group in dfg:
    print(type(group_key)) # groupby的值
    print(type(group)) # DataFrame
```
1. dfg可用的函数: get_group(group_key), size(), __get_item__(columns) 最后一个是只保留目标columns的聚合结果
2. dfg可用的属性: .groups/.indices: Dict[group_key: List[val_idx]]

##### 1.5 dfg.apply(func, include_groups=True)  按组处理

1. func(small_df)->ser/v
2. small_df是否有group_key这一个col就看include_groups, 一般设置成False是推荐的

##### 2. dfg.agg(**kwarg)  命名聚合
```python
res = df.groupby('User_ID').agg(
    Total_Amount=('Amount', 'sum'),  # new_col_name=('col', 'agg_method')
    Max_Amount=('Amount', 'max'),
    Last_Active_Date=('Date', 'max'),
    Unique_Items_Bought=('Item_ID', 'nunique')
) 
## res 是一个 name为User_ID，index变为组名的DataFrame
res = res.reset_index() # 将聚合DataFrame展平为原来的样子
```

##### 3. dfg[col].transform(func)  组内单列特征计算，返回结果与dfg[col]同形状
一般都先指定单列col，通过transform得到有ser用来替换原ser。
```python
df.groupby('User_ID')['age'].transform('max'/ func) # func(Ser)->Ser/v  func接受单组的整个Ser, 返回同形状的Ser就会对应位置更改, 返回单个值就是广播
# 你可以尝试在func直接返回Ser和一个值的区别
```
注意区分
```python 
res1 = dfg["滚木"].transform('max')
res2 = dfg.agg(
    Max_滚木=("滚木", "max")
).reset_index()

"""
res1 同形状 Series:
                    0    9    
                    1    9    
                    2    1    
                    3    9     
                    4    9     
                    5    1
res2 聚合结果 DataFrame:  
                        吉吉国王  Max_滚木
                    0     1         9
                    1     4         1
                    2     5         9
"""
```

##### 4. df.rolling(window: int, min_periods: int, closed: Literal['right','left','both','neither'])
返回Rolling对象，必须搭配后续操作才起作用。不过rolling的参数指定了计算范围
1. window表示窗口大小，min_periods表示边界是否padding, closed决定了窗口的尾部在哪里。left表示不包括当前行的前几行, right是默认的,表示包括当前的
2. Rolling.apply(func)   func(ser)->v  func的输入必然是某一列的窗口ser, 返回值一定是单值v

##### 5. pd.merge(left, right, how, on) / pd.merge(left, right, how, left_on, right_on) # 两种表连接方案
1. left, right 分别是表
2. how: Literal["inner", "left", "right", "outer"] 表示连接方式，left左连接就是假设on的col发生left表的主键在right表没找到，那不要删掉。inner表示都删掉
3. on: 如果左右表的主键col名相同直接用
4. left_on/right_on 不同就用这个
5. df.join() 是简化的左连接


##### 6. df.sort_values(by, ascending)
1. by        =col  or List[col]
2. ascending =Bool or List[Bool]