from pathlib import Path
from .constents import RAW_DATASET_BASE_DIR, BOLE_DATASET_BASE_DIR
CROSS_DOMAIN_AMAZON =  RAW_DATASET_BASE_DIR / "Amazon"
AMAZON_BOOKS_DIR =  CROSS_DOMAIN_AMAZON / "AmazonBooks"
AMAZON_BOOKS_INTER =  AMAZON_BOOKS_DIR / "AmazonBooks.inter"
AMAZON_BOOKS_ITEM =  AMAZON_BOOKS_DIR / "AmazonBooks.item"
AMAZON_MOVIES_DIR =  CROSS_DOMAIN_AMAZON / "AmazonMov"
AMAZON_MOVIES_INTER =  AMAZON_MOVIES_DIR / "AmazonMov.inter"
AMAZON_MOVIES_ITEM =  AMAZON_MOVIES_DIR / "AmazonMov.item"
import pandas as pd

def analyze_cross_domain_data():
   # 1. 读取数据 (只读取必要的列，假设第一列是 user_id, 第二列是 item_id)
   # RecBole 格式通常第一行是 header: user_id:token, item_id:token
   print("正在加载数据...")
   df_books = pd.read_csv(AMAZON_BOOKS_INTER, sep='\t', header=0)
   df_movies = pd.read_csv(AMAZON_MOVIES_INTER, sep='\t', header=0)

   # 提取列名 (RecBole 格式通常带类型后缀，如 user_id:token)
   book_user_col = [c for c in df_books.columns if 'user_id' in c][0]
   movie_user_col = [c for c in df_movies.columns if 'user_id' in c][0]

   # 2. 基础统计
   books_users = set(df_books[book_user_col].unique())
   movies_users = set(df_movies[movie_user_col].unique())

   print(f"=== 域 A: Books ===")
   print(f"交互数: {len(df_books)}")
   print(f"用户数: {len(books_users)}")
   print(f"物品数: {df_books.iloc[:, 1].nunique()}")

   print(f"\n=== 域 B: Movies ===")
   print(f"交互数: {len(df_movies)}")
   print(f"用户数: {len(movies_users)}")
   print(f"物品数: {df_movies.iloc[:, 1].nunique()}")

   # 3. 核心：重叠用户分析
   overlapping_users = books_users.intersection(movies_users)
   n_overlap = len(overlapping_users)

   print(f"\n=== 跨域重叠分析 (关键指标) ===")
   print(f"重叠用户数量: {n_overlap}")
   print(f"Books 域用户重叠率: {n_overlap / len(books_users):.2%}")
   print(f"Movies 域用户重叠率: {n_overlap / len(movies_users):.2%}")

   # 4. 决策建议
   if n_overlap < 1000:
      print("\n[警告] 重叠用户过少！很难训练跨域映射关系。")
   elif n_overlap / len(movies_users) < 0.1:
      print("\n[提示] 重叠率较低，这更像是一个 Zero-shot / Cold-start 任务。")
      print("建议：使用 Flow Matching 生成策略是非常合适的，因为你需要无中生有。")
   else:
      print("\n[完美] 重叠数据丰富，适合做训练。")


def get_column_name(df, prefix):
   """自动识别带有类型后缀的列名，例如 item_id:token"""
   cols = [c for c in df.columns if c.startswith(prefix)]
   if not cols:
      raise ValueError(f"Columns starting with {prefix} not found!")
   return cols[0]


def iterative_k_core(df, k, domain_name="Domain"):
   """
   对交互数据进行迭代式 K-Core 过滤
   """
   user_col = get_column_name(df, 'user_id')
   item_col = get_column_name(df, 'item_id')

   print(f"[-] [{domain_name}] Starting {k}-Core filtering...")
   print(f"    Initial: Users={df[user_col].nunique()}, Items={df[item_col].nunique()}, Interactions={len(df)}")

   epoch = 0
   while True:
      epoch += 1
      # 1. Filter Items
      item_counts = df[item_col].value_counts()
      valid_items = item_counts[item_counts >= k].index
      df = df[df[item_col].isin(valid_items)]

      # 2. Filter Users
      user_counts = df[user_col].value_counts()
      valid_users = user_counts[user_counts >= k].index
      df = df[df[user_col].isin(valid_users)]

      if len(df) == 0:
         print(f"[Warning] {domain_name} became empty after filtering!")
         return df

      # Check convergence
      # 如果过滤后的数量和过滤前判定的一致，说明稳定了
      curr_item_counts = df[item_col].value_counts()
      curr_user_counts = df[user_col].value_counts()

      if (curr_item_counts.min() >= k) and (curr_user_counts.min() >= k):
         break

   print(f"    Finished in {epoch} rounds.")
   print(f"    Final: Users={df[user_col].nunique()}, Items={df[item_col].nunique()}, Interactions={len(df)}")
   return df


def filter_item_metadata(inter_df, item_df):
   """
   根据过滤后的交互数据，同步过滤 .item 文件
   """
   if item_df is None:
      return None

   inter_item_col = get_column_name(inter_df, 'item_id')
   meta_item_col = get_column_name(item_df, 'item_id')

   # 获取交互数据中剩余的所有有效 item_id
   valid_items = set(inter_df[inter_item_col].unique())

   # 过滤 item_df
   filtered_item_df = item_df[item_df[meta_item_col].isin(valid_items)].copy()
   print(f"    Item Metadata filtered: {len(item_df)} -> {len(filtered_item_df)}")
   return filtered_item_df

def get_output_dir(core_k=5):
   return BOLE_DATASET_BASE_DIR / f"Amazon_Processed_{core_k}Core"

def convert_amazon_data(core_k=5):
   """
   主处理函数
   """
   # 1. 定义输入路径 (基于你的原始路径结构)
   # 请根据你实际的文件名修改这里的字符串
   # 2. 读取数据
   print("Loading raw datasets...")
   df_s_inter = pd.read_csv(AMAZON_BOOKS_INTER, sep='\t')
   df_t_inter = pd.read_csv(AMAZON_MOVIES_INTER, sep='\t')

   # 尝试读取 item 文件，如果不存在则为 None
   df_s_item = pd.read_csv(AMAZON_BOOKS_ITEM, sep='\t') if AMAZON_BOOKS_ITEM.exists() else None
   df_t_item = pd.read_csv(AMAZON_MOVIES_ITEM, sep='\t') if AMAZON_MOVIES_ITEM.exists() else None

   # 3. K-Core 过滤 (分别在各自域内进行)
   df_s_inter = iterative_k_core(df_s_inter, core_k, "Books")
   df_t_inter = iterative_k_core(df_t_inter, core_k, "Movies")

   # 4. 提取重叠用户 (Cross-Domain Intersection)
   s_user_col = get_column_name(df_s_inter, 'user_id')
   t_user_col = get_column_name(df_t_inter, 'user_id')

   s_users = set(df_s_inter[s_user_col].unique())
   t_users = set(df_t_inter[t_user_col].unique())

   overlap_users = s_users.intersection(t_users)
   print(f"\n[=== Cross-Domain Overlap Analysis ===]")
   print(f"Overlap Users (Both > {core_k}-core): {len(overlap_users)}")

   if len(overlap_users) == 0:
      print("[Error] No overlapping users found! Try reducing core_k.")
      return

   # 5. 生成最终交互数据 (只保留重叠用户)
   print("Filtering interactions to keep only overlapping users...")
   df_s_final = df_s_inter[df_s_inter[s_user_col].isin(overlap_users)].copy()
   df_t_final = df_t_inter[df_t_inter[t_user_col].isin(overlap_users)].copy()

   # 6. 同步过滤 Item Metadata 文件
   # 注意：Item 文件不需要只保留重叠用户交互过的物品，
   # 但通常为了节省空间，我们会过滤掉那些在 final_inter 中完全没出现过的物品
   print("Filtering item metadata files...")
   df_s_item_final = filter_item_metadata(df_s_final, df_s_item)
   df_t_item_final = filter_item_metadata(df_t_final, df_t_item)

   # 7. 保存结果
   # 输出目录：例如 dataset/Amazon_Processed_10Core/
   output_base = get_output_dir(core_k)

   # 即使是处理后的数据，RecBole 也建议用子文件夹存放
   # 结构:
   # output_base/
   #   Source_Books/
   #     Source_Books.inter
   #     Source_Books.item
   #   Target_Movies/
   #     Target_Movies.inter
   #     Target_Movies.item

   out_s_dir = output_base / "Source_Books"
   out_t_dir = output_base / "Target_Movies"

   out_s_dir.mkdir(parents=True, exist_ok=True)
   out_t_dir.mkdir(parents=True, exist_ok=True)

   print(f"\nSaving processed data to: {output_base}")

   # 保存 Books
   df_s_final.to_csv(out_s_dir / "Source_Books.inter", sep='\t', index=False)
   if df_s_item_final is not None:
      df_s_item_final.to_csv(out_s_dir / "Source_Books.item", sep='\t', index=False)

   # 保存 Movies
   df_t_final.to_csv(out_t_dir / "Target_Movies.inter", sep='\t', index=False)
   if df_t_item_final is not None:
      df_t_item_final.to_csv(out_t_dir / "Target_Movies.item", sep='\t', index=False)

   print("[Done] Data preparation complete.")
