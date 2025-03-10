import pandas as pd
import numpy as np


def print_all_countries_data(df: pd.DataFrame, sport: str, pre_years: list, post_years: list):
    """
    打印指定运动项目所有国家在特定时间段的数据
    """
    # 选择相关数据
    sport_data = df[df['Sport'] == sport].copy()

    # 计算每个国家每年的奖牌数
    medals_by_year = sport_data.groupby(['Year', 'NOC']).agg({
        'Medal': lambda x: sum(x != 'No medal')
    }).reset_index()

    # 创建完整的年份-国家组合
    study_years = pre_years + post_years
    all_countries = medals_by_year['NOC'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [study_years, all_countries],
        names=['Year', 'NOC']
    ).to_frame(index=False)

    # 与实际数据合并，缺失值填充为0
    medals_by_year = pd.merge(
        all_combinations,
        medals_by_year,
        how='left',
        on=['Year', 'NOC']
    ).fillna(0)

    # 创建透视表
    pivot_data = medals_by_year.pivot_table(
        values='Medal',
        index='NOC',
        columns='Year',
        aggfunc='sum',
        fill_value=0
    )

    # 计算每个阶段的平均值
    pivot_data['pre_mean'] = pivot_data[pre_years].mean(axis=1)
    pivot_data['post_mean'] = pivot_data[post_years].mean(axis=1)
    pivot_data['change'] = pivot_data['post_mean'] - pivot_data['pre_mean']

    # 筛选出至少在一个时期有奖牌的国家
    active_countries = pivot_data[
        (pivot_data['pre_mean'] > 0) |
        (pivot_data['post_mean'] > 0)
        ]

    # 打印结果
    print(f"\n{sport} Medal Counts by Country:")
    print("\nColumns:")
    print(f"- {pre_years}: Pre-treatment years")
    print(f"- {post_years}: Post-treatment years")
    print("- pre_mean: Average medals in pre-treatment period")
    print("- post_mean: Average medals in post-treatment period")
    print("- change: Difference between post and pre means")
    print("\nData (sorted by pre-treatment mean):")
    print(active_countries.sort_values('pre_mean', ascending=False).round(2))


def main():
    # 读取数据
    athletes_df = pd.read_csv("../summerOly_athletes.csv")

    # 排球分析时期
    print("\n=== Volleyball (2000-2012) ===")
    print_all_countries_data(
        athletes_df,
        'Volleyball',
        [2000, 2004],
        [2008, 2012]
    )

    # 体操分析时期
    print("\n=== Gymnastics (1976-1988) ===")
    print_all_countries_data(
        athletes_df,
        'Gymnastics',
        [1976, 1980],
        [1984, 1988]
    )


if __name__ == "__main__":
    main()