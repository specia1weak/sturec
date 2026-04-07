在序列推荐里面，recbole不会给rating这一列剔除掉低评分的样本。关于rating的处理方式
1. 序列推荐：直接剔除
```yaml
val_interval:
  rating: "[3, 5]"  # 只保留评分在 3 到 5 之间的交互记录
```
2. 点对点推荐：使用threshold映射低评分0，高评分1
```yaml
threshold:
    rating: 4.0
```