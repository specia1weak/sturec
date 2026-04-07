import polars as pl
import tempfile
import os

# ==========================================
# 准备工作：在本地生成一个测试用的 CSV 文件
# ==========================================
temp_dir = tempfile.gettempdir()
file_path = os.path.join(temp_dir, "mock_market_data.csv")

# 随便造一点 A 股的模拟数据并写入 CSV
mock_data = pl.DataFrame({
    "ticker": ["SH600519", "SH600519", "SZ000858", "SZ000858", "SH600036", "SH600036"],
    "date": ["2026-03-01", "2026-03-02", "2026-03-01", "2026-03-02", "2026-03-01", "2026-03-02"],
    "price": [1700.5, 1710.0, 150.2, 148.5, 30.5, 31.2],
    "volume": [1000, 1500, 5000, 4800, 10000, 12000]
})
mock_data.write_csv(file_path)
print(f"✅ 步骤 1: 模拟的大型 CSV 文件已生成于 {file_path}\n")

# ==========================================
# 核心教学：如何使用 Polars 的机制处理数据
# ==========================================

# 机制一：惰性扫描 (Lazy Scanning)
# 使用 scan_csv 而不是 read_csv。此时 Polars 仅仅是记住了文件路径和列名，完全没有把数据读入内存。
lazy_plan = pl.scan_csv(file_path)
print("🔍 步骤 2: lazy_plan 已创建。此时内存占用几乎为 0，因为数据还没读取。")
print(f"变量类型: {type(lazy_plan)}\n")

# 机制二：上下文与表达式 (Contexts & Expressions)
# 我们开始向 Polars 下达处理逻辑。注意 pl.col()，这就是“表达式”。
# 我们只是在编排逻辑，这一步依然没有真正处理数据。
query = (
    lazy_plan
    # 1. Filter 上下文：过滤数据
    .filter(pl.col("ticker").is_in(["SH600519", "SZ000858"]))

    # 2. Select / With_columns 上下文：增加新列 (价格 * 交易量 = 交易额)
    .with_columns(
        (pl.col("price") * pl.col("volume")).alias("turnover")
    )

    # 3. Groupby 上下文：按股票代码分组，并计算平均交易额
    .group_by("ticker")
    .agg(
        pl.col("turnover").mean().alias("avg_turnover")
    )
)

print("⚙️ 步骤 3: 逻辑查询计划已经构建完毕。")

# 机制三：触发执行 (Execution & Streaming)
# 只有调用 collect() 时，底层的 Rust 引擎才开始真正干活。
# 参数 streaming=True 告诉引擎：像流水线一样一小块一小块地处理，别把内存撑爆。
print("🚀 步骤 4: 开始执行流式计算...\n")
final_result = query.collect(engine="streaming")

print("📊 最终结果输出：")
print(final_result)
print(type(final_result))

# 清理测试文件
os.remove(file_path)