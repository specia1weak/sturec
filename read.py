from pathlib import Path

from src.convert.amazon import AmazonBooksLoader, AmazonCrossDomainLoader

if __name__ == "__main__":
    # 1. 定义输出路径
    output_path = Path("dataset/Amazon-5core")

    # 2. 实例化并运行处理流程 (自带 5-core 过滤)
    loader = AmazonCrossDomainLoader(filter_k_core=5)
    loader.export_to_recbole_format(df_type="inter", to_csv_path="dataset/Amazon-5core/amazon.inter")