```python
class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])
```
-- --

### Dataloader机制
```python
DataLoader(dataset, batch_size=4, shuffle=True)
```
1. shuffle=True: RandomSampler生成随机序列[10, 2, 5, 8, ...]
2. batch_size=4: BatchSampler包装RandomSampler一次生成一batchsize
3. num_workers > 0: 多进程处理


#### 一、多进程机制
1. 生产者Sampler: 派发任务（大小为batch_size的索引数组，等待变成tensor)
2. workers: 进程池，每个worker有自己的任务队列，被Sampler塞任务，处理完扔给输出队列
3. 输出队列: 只有一个，被for batch in dataloader消费

#### 二、整理函数collate_fn
1. 每个worker通过collect_fn自定义处理流程
2. def collate_fn(List[Dataset.__get_item__[idx]]) -> Any
```python
samples = [dataset[i] for i in indices] # Worker 获取单条数据
batch = collate_fn(samples)             # Worker 调用打包函数
yield batch
---------
for batch in dataloader
```

