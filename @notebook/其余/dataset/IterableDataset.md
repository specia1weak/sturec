## torch: IterableDataset
1. 避免Map-style Dataset对于__get_item__实现的要求，不必一次性读取所有数据，压力小
2. 代价1：是无法全局Shuffle以及，想要读到后面的东西必须不能中断
3. 代价2：你必须显式管理多进程worker，他们都是从头开始读取的，不管理你会拿到一样的数据
```python
class IterableDataset(Dataset[T_co])
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])
```
-- --
### Dataloader针对机制
```python
if isinstance(dataset, IterableDataset):

```