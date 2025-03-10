import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
def did_analysis(df: pd.DataFrame, treatment_country: str, control_countries: list,
                 pre_years: list, post_years: list, sport: str, coach_name: str):
    """
    进行DID分析

    Parameters:
    -----------
    df : DataFrame
        运动员数据
    treatment_country : str
        实验组国家
    control_countries : list
        对照组国家列表
    pre_years : list
        干预前年份列表
    post_years : list
        干预后年份列表
    sport : str
        运动项目
    coach_name : str
        教练姓名
    """
    print(f"\n{'=' * 20} {sport} Analysis {'=' * 20}")
    print(f"Analyzing the effect of {coach_name} on {treatment_country}'s {sport} team")

    # 数据准备
    sport_data = df[df['Sport'] == sport].copy()
    medals_by_year = sport_data.groupby(['Year', 'NOC']).agg({
        'Medal': lambda x: sum(x != 'No medal')
    }).reset_index()

    # 创建完整数据集
    study_years = pre_years + post_years
    all_combinations = pd.MultiIndex.from_product([
        study_years,
        [treatment_country] + control_countries
    ], names=['Year', 'NOC']).to_frame(index=False)

    # 合并数据
    did_data = pd.merge(
        all_combinations,
        medals_by_year,
        how='left',
        on=['Year', 'NOC']
    ).fillna(0)

    # 添加DID变量
    did_data['treated'] = (did_data['NOC'] == treatment_country).astype(int)
    did_data['post'] = did_data['Year'].isin(post_years).astype(int)
    did_data['did'] = did_data['treated'] * did_data['post']

    # 描述性统计
    print("\nMedal Counts by Country and Year:")
    pivot_table = did_data.pivot_table(
        values='Medal',
        index='NOC',
        columns='Year',
        aggfunc='sum'
    ).round(2)
    print(pivot_table)

    # 计算平均效应
    means = did_data.groupby(['treated', 'post'])['Medal'].mean().unstack()
    print("\nMean Medals by Period:")
    print("Pre-treatment period:", pre_years)
    print("Post-treatment period:", post_years)
    print(means)

    # 计算原始DID估计
    did_estimate = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
    print(f"\nRaw DID Estimate: {did_estimate:.2f} medals")

    # 回归分析
    X = sm.add_constant(did_data[['treated', 'post', 'did']])
    y = did_data['Medal']
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC1')

    # 可视化平行趋势
    plt.figure(figsize=(12, 6))

    # 处理组趋势
    treatment_data = did_data[did_data['treated'] == 1]
    plt.plot(treatment_data.groupby('Year')['Medal'].mean(),
             marker='o', label=f'{treatment_country} (Treatment)',
             linewidth=2)

    # 控制组趋势
    control_data = did_data[did_data['treated'] == 0]
    plt.plot(control_data.groupby('Year')['Medal'].mean(),
             marker='s', label='Control Group Average',
             linewidth=2)

    # 添加垂直线表示干预时点
    intervention_year = (pre_years[-1] + post_years[0]) / 2
    plt.axvline(x=intervention_year, color='r', linestyle='--',
                label='Intervention')

    plt.title(f'Medal Trends in {sport}\n{coach_name} Effect Analysis',
              fontsize=14, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Medals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sport}_{coach_name}_effect.png')
    plt.close()

    return results, did_estimate, means


def interpret_results(results, sport: str, coach_name: str):
    """解释DID分析结果"""
    # 获取系数和标准误
    did_coef = results.params['did']
    did_se = results.bse['did']

    # 计算置信区间
    ci_95 = stats.t.interval(0.95, results.df_resid,
                             loc=did_coef,
                             scale=did_se)

    print(f"\n{'=' * 20} Results Interpretation {'=' * 20}")
    print(f"\nEffect of {coach_name} on {sport}:")
    print(f"Estimated effect: {did_coef:.2f} medals")
    print(f"95% Confidence Interval: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
    print(f"P-value: {results.pvalues['did']:.4f}")

    # 解释效应
    if results.pvalues['did'] < 0.05:
        if did_coef > 0:
            print(f"\nThe effect is POSITIVE and SIGNIFICANT")
            print(f"We can be 95% confident that {coach_name} increased")
            print(f"the medal count by between {ci_95[0]:.1f} and {ci_95[1]:.1f} medals")
        else:
            print(f"\nThe effect is NEGATIVE and SIGNIFICANT")
    else:
        print(f"\nThe effect is NOT STATISTICALLY SIGNIFICANT")
        print("We cannot conclude there was a meaningful effect")


def main():
    # 读取数据
    athletes_df = pd.read_csv("../summerOly_athletes.csv")

    # 分析郎平效应
    volleyball_results, v_estimate, v_means = did_analysis(
        athletes_df,
        'USA',
        ['ITA', 'CHN'],
        [2000, 2004],
        [2008, 2012],
        'Volleyball',
        'Lang Ping'
    )
    interpret_results(volleyball_results, 'Volleyball', 'Lang Ping')

    # 分析Károlyi效应
    gymnastics_results, g_estimate, g_means = did_analysis(
        athletes_df,
        'USA',
        ['ROU', 'JPN'],
        [1976, 1980],
        [1984, 1988],
        'Gymnastics',
        'Béla Károlyi'
    )
    interpret_results(gymnastics_results, 'Gymnastics', 'Béla Károlyi')


if __name__ == "__main__":
    main()