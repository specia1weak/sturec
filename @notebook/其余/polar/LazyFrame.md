## LazyFrame
1. 底层是计算图（DAG）直到触发调用函数。
2. 通过pl.scan_csv() 或者pl.lazy()得到

```python
def scan_csv(
    source: str | Path,
    *,
    has_header: bool = True,
    separator: str = ",",
    schema: dict[str, DataType] | Schema | None = None,  # 省去自动推理 {"ticker": pl.Utf8, "price": pl.Float64}
    ignore_errors: bool = False,
    ...
) -> LazyFrame
```


## 算子(Transformations) lf->lf
### 零、Expr是什么
1. Expr是操作指令, 往往起源于pl.col("name")
2. Expr有名字，pl.col("price") * pl.col("volume")默认有名字【(col("price") * col("volume"))】 \
   .alias("avg_price")改名字
3. 不区分ScalarExpr 或 ListExpr
4. 基础算子：+, -, *, /, **, >, <, ==, is_in(), is_not_null()
5. 条件运算when(BoolExpr).then(Expr).otherwise(Expr)
6. 缺失值.fill_null(strategy="")


### 一、上下文Contxt
1. select(*Expr): 只留下指定Expr列
2. with_columns(*Expr): 新增/update Expr列
3. filter(*BoolExpr): 留下BoolExpr对应的行 filter(pl.col("price") > 10)
4. group_by("col" | Expr, maintain_order): 返回LazyGroupBy必须后接agg \
   .agg(*Expr):  \
   .len(): 可以替代agg，相当于SQL的 `count(*)`
5. sort([])

### 二、over()最后指定范围
1. .over()总是出现在最后，虽然有点反直觉，但是polars就是这样解析的

### 三、多表
1. join(lf2, on="", how)
2. pl.concat([lf1, lf2])

### 四、ser->val缩减
1. 统计：sum()/mean()/max/min()/std/var()/median()/quantile(0.95) # 95%的值小于等于返回值
2. 计数：count()/n_unique()/null_count()/any()/all()
3. 位置：first()/last()
4. implode(): 将整列 $N$ 个元素聚合成一个包含 $N$ 个元素的 List

### 五、展开explode与收集implode
1. explode() 将List拆行
2. implode 将多行数据合并成单List

### 六、特殊
1. shift(1): 整体下移，第一个填充null，最后一个直接消失
2. rolling_list(n): 包括本行，拿取n个历史值
3. 
## 触发(Actions) lf->df
1. collect(engine="cpu"|"streaming")
2. count()
3. show()