import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def parallel_trends_test(data: pd.DataFrame, pre_years: list):
    """
    检验平行趋势假设
    """
    # 只使用处理前的数据
    pre_data = data[data['Year'].isin(pre_years)].copy()

    # 计算处理组和对照组的趋势
    trends = pre_data.groupby(['treated', 'Year'])['Medal'].mean().unstack()

    # 计算每组的变化率
    if len(pre_years) >= 2:
        treatment_change = (trends.loc[1, pre_years[1]] - trends.loc[1, pre_years[0]])
        control_change = (trends.loc[0, pre_years[1]] - trends.loc[0, pre_years[0]])

        # 使用t检验比较两组的变化
        t_stat, p_value = stats.ttest_ind(
            [treatment_change] * len(pre_data[pre_data['treated'] == 1]),
            [control_change] * len(pre_data[pre_data['treated'] == 0])
        )

        print("\nParallel Trends Test:")
        print(f"Treatment group change: {treatment_change:.2f}")
        print(f"Control group change: {control_change:.2f}")
        print(f"t-statistic: {t_stat:.2f}")
        print(f"p-value: {p_value:.4f}")

        # 可视化趋势
        plt.figure(figsize=(10, 6))
        plt.plot(pre_years, trends.loc[1], marker='o', label='Treatment Group')
        plt.plot(pre_years, trends.loc[0], marker='o', label='Control Group')
        plt.title('Pre-treatment Trends')
        plt.xlabel('Year')
        plt.ylabel('Average Medals')
        plt.legend()
        plt.grid(True)
        plt.show()

        return p_value > 0.05  # 返回是否满足平行趋势假设
    return None



def effect_significance_tests(results):
    """
    进行处理效应的统计显著性检验
    """
    # 获取DID估计量及其标准误
    did_coef = results.params['did']
    did_se = results.bse['did']

    # 1. t检验
    t_stat = did_coef / did_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), results.df_resid))

    # 2. 置信区间
    ci_95 = stats.t.interval(0.95, results.df_resid,
                             loc=did_coef,
                             scale=did_se)

    print("\nTreatment Effect Significance Tests:")
    print(f"1. t-test:")
    print(f"   t-statistic: {t_stat:.2f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"2. 95% Confidence Interval: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")

    # 3. F检验（联合显著性）
    f_test = results.f_test('did = 0')
    print(f"3. F-test:")
    print(f"   F-statistic: {f_test.statistic:.2f}")
    print(f"   p-value: {f_test.pvalue:.4f}")


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