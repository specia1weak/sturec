# run-amazon.py
from recbole.quick_start import run_recbole

def run_experiment():
    """
    运行 Amazon 数据集实验
    支持 DeepFM, MMoE, DCN 等 CTR 预测模型
    """
    # 这里定义的参数优先级 > yaml 文件
    # 可以在这里方便地调整超参，不用反复改文件
    parameter_dict = {
        # ===========================
        # 1. 硬件与环境配置
        # ===========================
        'use_gpu': True,          # 开启 GPU 训练
        'gpu_id': 0,              # 指定显卡 ID，如果你是单卡通常是 0
        'seed': 2020,             # 固定随机种子，保证结果可复现
        'state': 'INFO',          # 日志级别 (INFO/WARNING/ERROR)

        # ===========================
        # 2. 数据集配置
        # ===========================
        # Amazon 数据集有 3 个 domain，数据量较大 (80万+交互)
        # 注意：不要过滤，否则会导致 user/item 文件为空
        'min_user_inter_num': 0,
        'min_item_inter_num': 0,
        
        # ===========================
        # 3. 训练超参数
        # ===========================
        'epochs': 30,             # 最大训练轮数
        'train_batch_size': 4096, # 训练批次大小 (Amazon数据量大，可以用大一些的batch)
        'learner': 'adam',        # 优化器
        'learning_rate': 0.001,   # 学习率
        'eval_step': 1,           # 每训练几轮验证一次
        'stopping_step': 10,      # 早停机制：如果验证集指标 10 轮不涨就停止

        # ===========================
        # 4. 模型特定参数
        # ===========================
        # 如果你的 yaml 里已经写了，这里可以不写；写了会覆盖 yaml
        'embedding_size': 16,           # 嵌入维度
        'mlp_hidden_size': [128, 64],   # MLP 隐藏层大小
        'dropout_prob': 0.2,            # Dropout 比例

        # ===========================
        # 5. 评估设置 (CTR 任务)
        # ===========================
        # CTR 点击率预估通常看 AUC 和 LogLoss
        # 注意：不能同时使用 Ranking metrics (Precision/Recall) 和 Value metrics (AUC/LogLoss)
        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',    # 选取模型时主要看哪个指标（AUC越高越好）
        'eval_batch_size': 8192,  # 评估时不需要反向传播，批次可以大一点加快速度
        
        # ===========================
        # 6. 保存与日志配置
        # ===========================
        'checkpoint_dir': 'saved',       # 模型保存路径
        'show_progress': True,            # 显示训练进度条
    }

    # ===========================
    # 7. 开始运行
    # ===========================
    # model: 可以选择 'DeepFM', 'MMoE', 'DCN', 'WideDeep', 'xDeepFM' 等
    # dataset: 'amazon' (对应 dataset/amazon/ 文件夹)
    # config_file_list: 加载 yaml 配置文件
    
    print("="*80)
    print("Amazon 数据集 - CTR 预测实验")
    print("="*80)
    print(f"模型: DeepFM")
    print(f"数据集: Amazon (Beauty + Clothing + Health)")
    print(f"GPU: {'开启' if parameter_dict['use_gpu'] else '关闭'}")
    print(f"训练轮数: {parameter_dict['epochs']}")
    print(f"Batch Size: {parameter_dict['train_batch_size']}")
    print("="*80)
    
    run_recbole(
        model='DeepFM',                              # 使用 DeepFM 模型
        dataset='amazon',                            # 数据集名称
        config_file_list=['dataset/amazon/amazon.yaml'],  # 配置文件路径
        config_dict=parameter_dict                   # 超参数字典
    )
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

def run_mmoe_experiment():
    """
    使用 MMoE 模型运行 Amazon 数据集实验
    MMoE 适合多任务学习，可以同时预测点击和其他任务
    """
    parameter_dict = {
        'use_gpu': True,
        'gpu_id': 0,
        'seed': 2020,
        'state': 'INFO',
        
        'min_user_inter_num': 5,
        'min_item_inter_num': 5,
        
        'epochs': 30,
        'train_batch_size': 4096,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,
        'stopping_step': 10,
        
        # MMoE 特定参数
        'embedding_size': 16,
        'mlp_hidden_size': [128, 64],
        'dropout_prob': 0.2,
        'num_experts': 4,           # 专家数量
        'expert_hidden_size': [128, 64],  # 每个专家的隐藏层
        
        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',
        'eval_batch_size': 8192,
        'checkpoint_dir': 'saved',
        'show_progress': True,
    }
    
    print("="*80)
    print("Amazon 数据集 - MMoE 多任务学习实验")
    print("="*80)
    
    run_recbole(
        model='MMoE',
        dataset='amazon',
        config_file_list=['dataset/amazon/amazon.yaml'],
        config_dict=parameter_dict
    )

def run_dcn_experiment():
    """
    使用 DCN (Deep & Cross Network) 模型
    DCN 擅长捕捉特征交叉
    """
    parameter_dict = {
        'use_gpu': True,
        'gpu_id': 0,
        'seed': 2020,
        'state': 'INFO',
        
        'min_user_inter_num': 5,
        'min_item_inter_num': 5,
        
        'epochs': 30,
        'train_batch_size': 4096,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,
        'stopping_step': 10,
        
        # DCN 特定参数
        'embedding_size': 16,
        'mlp_hidden_size': [128, 64],
        'cross_layer_num': 3,       # Cross 网络层数
        'dropout_prob': 0.2,
        
        'metrics': ['AUC', 'LogLoss'],
        'valid_metric': 'AUC',
        'eval_batch_size': 8192,
        'checkpoint_dir': 'saved',
        'show_progress': True,
    }
    
    print("="*80)
    print("Amazon 数据集 - DCN 实验")
    print("="*80)
    
    run_recbole(
        model='DCN',
        dataset='amazon',
        config_file_list=['dataset/amazon/amazon.yaml'],
        config_dict=parameter_dict
    )

if __name__ == '__main__':
    # 选择要运行的实验
    # 方式1: 直接运行 DeepFM
    # run_experiment()
    
    # 方式2: 运行 MMoE (取消注释下面的代码)
    run_mmoe_experiment()
    
    # 方式3: 运行 DCN (取消注释下面的代码)
    # run_dcn_experiment()
    
    # 方式4: 对比多个模型（依次运行）
    # print("\n运行 DeepFM...")
    # run_experiment()
    # print("\n运行 DCN...")
    # run_dcn_experiment()

