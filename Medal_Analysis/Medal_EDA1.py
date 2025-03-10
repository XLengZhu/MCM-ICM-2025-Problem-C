import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取所有数据集，指定编码
medal_counts = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')
hosts = pd.read_csv('../summerOly_hosts.csv', encoding='utf-8')
programs = pd.read_csv('../summerOly_programs.csv', encoding='cp1252')
athletes = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


# 对每个数据集进行基本信息查看
def check_dataset(df, name):
    print(f"\n{'-' * 50}")
    print(f"Dataset: {name}")
    print(f"{'-' * 50}")

    # 基本信息
    print("\n1. 基本信息:")
    print(f"Shape: {df.shape}")
    print("\nColumns:", list(df.columns))

    # 数据类型
    print("\n2. 数据类型:")
    print(df.dtypes)

    # 缺失值情况
    print("\n3. 缺失值统计:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

    # 基本统计描述
    if df.select_dtypes(include=[np.number]).columns.any():
        print("\n4. 数值列统计描述:")
        print(df.describe())
    else:
        print("\n4. 无数值列，跳过统计描述")

    # 查看唯一值数量
    print("\n5. 每列唯一值数量:")
    print(df.nunique())

    return None


# 检查每个数据集
print("开始数据探索分析...\n")
for df, name in [(medal_counts, "Medal Counts"),
                 (hosts, "Hosts"),
                 (programs, "Programs"),
                 (athletes, "Athletes")]:
    check_dataset(df, name)