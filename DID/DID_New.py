import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def did_analysis(df: pd.DataFrame, treatment_country: str, control_countries: list,
                 pre_years: list, post_years: list, sport: str):
    """
    进行DID分析
    """
    # 选择相关数据
    sport_data = df[df['Sport'] == sport].copy()

    # 计算每个国家每年的奖牌数
    medals_by_year = sport_data.groupby(['Year', 'NOC']).agg({
        'Medal': lambda x: sum(x != 'No medal')
    }).reset_index()

    # 创建完整的年份-国家组合
    all_combinations = pd.MultiIndex.from_product([
        pre_years + post_years,
        [treatment_country] + control_countries
    ], names=['Year', 'NOC']).to_frame(index=False)

    # 与实际数据合并，缺失值填充为0
    medals_by_year = pd.merge(
        all_combinations,
        medals_by_year,
        how='left',
        on=['Year', 'NOC']
    ).fillna(0)

    # 筛选数据
    study_years = pre_years + post_years
    did_data = medals_by_year[
        (medals_by_year['Year'].isin(study_years)) &
        ((medals_by_year['NOC'] == treatment_country) |
         (medals_by_year['NOC'].isin(control_countries)))
        ].copy()

    print(f"\nMedal counts by country and year:")
    pivot_table = did_data.pivot_table(
        values='Medal',
        index='NOC',
        columns='Year',
        aggfunc='sum'
    ).round(2)
    print(pivot_table)

    # 添加DID变量
    did_data['treated'] = (did_data['NOC'] == treatment_country).astype(int)
    did_data['post'] = did_data['Year'].isin(post_years).astype(int)
    did_data['did'] = did_data['treated'] * did_data['post']

    # 计算处理组和对照组在干预前后的平均值
    means = did_data.groupby(['treated', 'post'])['Medal'].mean().unstack()
    print("\nMean medals by group and period:")
    print(means)
    print("\nDifference-in-Differences:")
    did_estimate = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
    print(f"Raw DID estimate: {did_estimate:.2f}")

    # 回归分析
    X = sm.add_constant(did_data[['treated', 'post', 'did']])
    y = did_data['Medal']

    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC1')  # 使用稳健标准误

    print("\nRegression Results:")
    print(results.summary().tables[1])

    # 平行趋势检验
    pre_data = did_data[did_data['Year'].isin(pre_years)].copy()
    trends = pre_data.groupby(['treated', 'Year'])['Medal'].mean().unstack()

    if len(pre_years) >= 2:
        treatment_change = (trends.loc[1, pre_years[1]] - trends.loc[1, pre_years[0]])
        control_change = (trends.loc[0, pre_years[1]] - trends.loc[0, pre_years[0]])

        print("\nParallel Trends Test:")
        print(f"Treatment group change: {treatment_change:.2f}")
        print(f"Control group change: {control_change:.2f}")

    return results


def main():
    # 读取数据
    athletes_df = pd.read_csv("../summerOly_athletes.csv")

    # 排球分析 - 郎平(新对照组)
    print("\n=== Volleyball Analysis ===")
    volleyball_results = did_analysis(
        athletes_df,
        'USA',
        ['ITA', 'CHN'],  # 新的对照组
        [2000, 2004],  # 前两届
        [2008, 2012],  # 后两届
        'Volleyball'
    )

    # 体操分析 - Béla Károlyi(新对照组)
    print("\n=== Gymnastics Analysis ===")
    gymnastics_results = did_analysis(
        athletes_df,
        'USA',
        ['ROU', 'JPN'],  # 新的对照组
        [1976, 1980],  # 前两届
        [1984, 1988],  # 后两届
        'Gymnastics'
    )


if __name__ == "__main__":
    main()