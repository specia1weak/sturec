### 一、Config配置变量
```python
internal_config_dict = {'ENTITY_ID_FIELD': 'entity_id', 'HEAD_ENTITY_ID_FIELD': 'head_id', 'ITEM_ID_FIELD': 'item_id', 'ITEM_LIST_LENGTH_FIELD': 'item_length', 'LABEL_FIELD': 'label', 'LIST_SUFFIX': '_list', 'MAX_ITEM_LIST_LENGTH': 50, 'MODEL_TYPE': <ModelType.SEQUENTIAL: 2>, 'NEG_PREFIX': 'neg_', 'POSITION_FIELD': 'position_id', 'RATING_FIELD': 'rating', 'RELATION_ID_FIELD': 'relation_id', 'TAIL_ENTITY_ID_FIELD': 'tail_id', 'TIME_FIELD': 'timestamp', 'USER_ID_FIELD': 'user_id', 'additional_feat_suffix': None, 'alias_of_entity_id': None, 'alias_of_item_id': None, 'alias_of_relation_id': None, 'alias_of_user_id': None, 'benchmark_filename': None, 'checkpoint_dir': 'saved', 'clip_grad_norm': None, 'data_path': 'dataset/', 'dataloaders_save_path': None, 'dataset_save_path': None, 'discretization': None, 'dropout_prob': 0.3, 'embedding_size': 64, 'enable_amp': False, 'enable_scaler': False, 'entity_kg_num_interval': '[0,inf)', 'epochs': 300, 'eval_args': {'mode': 'full', 'order': 'TO', 'split': {'LS': 'valid_and_test'}}, 'eval_batch_size': 4096, 'eval_step': 1, 'field_separator': '	', 'filter_inter_by_user_or_item': True, 'gpu_id': '0', 'hidden_size': 128, 'item_inter_num_interval': '[0,inf)', 'kg_reverse_r': False, 'learner': 'adam', 'learning_rate': 0.001, 'load_col': {'inter': ['user_id', 'item_id']}, 'log_wandb': False, 'loss_decimal_place': 4, 'loss_type': 'CE', 'metric_decimal_place': 4, 'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'], 'normalize_all': None, 'normalize_field': None, 'num_layers': 1, 'numerical_features': [], 'preload_weight': None, 'relation_kg_num_interval': '[0,inf)', 'repeatable': True, 'reproducibility': True, 'require_pow': False, 'rm_dup_inter': None, 'save_dataloaders': False, 'save_dataset': False, 'seed': 2020, 'seq_len': None, 'seq_separator': ' ', 'show_progress': True, 'shuffle': True, 'state': 'INFO', 'stopping_step': 10, 'threshold': None, 'topk': [10], 'train_batch_size': 2048, 'train_neg_sample_args': {'alpha': 1.0, 'candidate_num': 0, 'distribution': 'uniform', 'dynamic': False, 'sample_num': 1}, 'transform': None, 'unload_col': None, 'unused_col': None, 'use_gpu': True, 'user_inter_num_interval': '[0,inf)', 'val_interval': None, 'valid_metric': 'MRR@10', 'valid_metric_bigger': True, 'wandb_project': 'recbole', 'weight_decay': 0.0, 'worker': 0}
external_config_dict = {'ITEM_ID_FIELD': 'item_id', 'LABEL_FIELD': 'label', 'MAX_ITEM_LIST_LENGTH': 10, 'RATING_FIELD': 'rating', 'TIME_FIELD': 'timestamp', 'USER_ID_FIELD': 'user_id', 'dataset': 'ml-1m', 'eval_args': {'mode': 'full', 'order': 'TO', 'split': {'LS': 'valid_and_test'}}, 'field_separator': '	', 'gpu_id': '', 'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'], 'user': ['user_id', 'age', 'gender', 'occupation']}, 'threshold': {'rating': 4.0}, 'train_neg_sample_args': None, 'val_interval': {'rating': '[4.0, 5.0]'}}
```

### 二、Config覆盖机制
合并顺序 internal默认配置-文件-参数-命令行
```python
def _merge_external_config_dict(self):
    external_config_dict = dict()
    external_config_dict.update(self.file_config_dict) # 传入的yaml文件列表
    external_config_dict.update(self.variable_config_dict) # 传入的函数参数字典
    external_config_dict.update(self.cmd_config_dict) # 命令行
    self.external_config_dict = external_config_dict

def _get_final_config_dict(self):
    final_config_dict = dict()
    final_config_dict.update(self.internal_config_dict)
    final_config_dict.update(self.external_config_dict)
    return final_config_dict
```