import os
import shutil

import duckdb
from typing import Union, List
from pathlib import Path


def duckdb_sort_parquet(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "8GB",  # 物理防爆盾：强制最高内存限制
        temp_dir: str = "./workspace/tmp"  # 磁盘防爆盾：防止把系统 C 盘写满
):
    input_path = str(input_path)
    output_path = str(output_path)

    # 确保安全的临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 统一转换 sort_by 和 descending 为列表，对齐 Polars 体验
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = []
    for col, desc in zip(sort_by, descending):
        direction = "DESC" if desc else "ASC"
        order_clauses.append(f'"{col}" {direction}')
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动核外大表物理排序...")
    print(f"    -> 输入: {input_path}")
    print(f"    -> 输出: {output_path}")
    print(f"    -> 规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存不超过 {memory_limit}, 临时缓冲: {temp_dir}")

    # 建立 DuckDB 连接并注入资源防爆指令
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")

    # 执行查询并流式落盘，开启高压缩比
    query = f"""
        COPY (
            SELECT * FROM read_parquet('{input_path}')
            ORDER BY {order_by_str}
        ) TO '{output_path}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
    """
    con.execute(query)
    con.close()

def sort_parquet_inplace(
        file_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "8GB",
        temp_dir = "./tmp"
):
    """
    使用 DuckDB 对 Parquet 文件进行极低内存的“原地排序”。
    机制：将排序结果输出到 ./tmp 目录下，完成后直接覆盖原文件，绝对安全。
    """
    file_path = Path(file_path).absolute()
    if not file_path.exists():
        raise FileNotFoundError(f"找不到目标文件: {file_path}")

    # 创建安全的临时目录
    temp_dir_path = Path(temp_dir).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_output_path = temp_dir_path / f"sorted_{file_path.name}"

    # 统一转换 sort_by 和 descending 为列表
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = [
        f'"{col}" {"DESC" if desc else "ASC"}'
        for col, desc in zip(sort_by, descending)
    ]
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动大表原地物理排序 (In-place Sort)...")
    print(f"    -> 目标文件: {file_path}")
    print(f"    -> 排序规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存 <= {memory_limit}, 临时引擎目录: {temp_dir_path}")

    try:
        # 建立 DuckDB 连接并注入资源限制
        con = duckdb.connect()
        # 注意：DuckDB 接收路径最好用 POSIX 格式 (特别是 Windows 下)，避免转义坑
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{temp_dir_path.as_posix()}'")

        # 执行查询并流式落盘到 temp_output_path
        query = f"""
            COPY (
                SELECT * FROM read_parquet('{file_path.as_posix()}')
                ORDER BY {order_by_str}
            ) TO '{temp_output_path.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
        """
        con.execute(query)
        con.close()

        # ✨ 核心：安全地替换原文件
        # 使用 shutil.move 可以跨磁盘移动，并且覆盖原文件
        shutil.move(str(temp_output_path), str(file_path))
        print(f"[DuckDB] ✅ 排序完成！已成功覆写原文件: {file_path.name}\n")

    except Exception as e:
        print(f"\n[DuckDB] ❌ 排序过程中发生错误: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise e