import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 读取数据
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')

# 2. 基尼系数分析
def calculate_gini(data):
    array = np.array(data)
    if np.any(array < 0):
        return None
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def analyze_medal_concentration():
    # 计算每年的基尼系数
    yearly_gini = df.groupby('Year').apply(
        lambda x: pd.Series({
            'Gini_Total': calculate_gini(x['Total']),
            'Gini_Gold': calculate_gini(x['Gold']),
            'Participating_Countries': len(x)
        })
    ).reset_index()

    print("\n2. Medal Concentration Analysis:")
    print("\nGini Coefficient by Period:")
    yearly_gini['Period'] = pd.cut(yearly_gini['Year'],
                                   bins=[1890, 1912, 1952, 1988, 2024],
                                   labels=['1896-1912', '1920-1952',
                                           '1956-1988', '1992-2024'])
    print(yearly_gini.groupby('Period')[['Gini_Total', 'Gini_Gold']].mean().round(3))

    # 可视化基尼系数趋势
    plt.figure(figsize=(15, 6))
    plt.plot(yearly_gini['Year'], yearly_gini['Gini_Total'], 'o-',
             label='Total Medals', color='blue')
    plt.plot(yearly_gini['Year'], yearly_gini['Gini_Gold'], 's-',
             label='Gold Medals', color='gold')

    plt.title('Gini Coefficient Trend in Olympic Medals')
    plt.xlabel('Year')
    plt.ylabel('Gini Coefficient')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_gini['Year'],rotation=45)
    plt.tight_layout()
    plt.savefig('gini_coefficient_trend.png', dpi=300, bbox_inches='tight')
    plt.close()


# 3. 排名与奖牌数关系分析
def analyze_rank_medal_relationship():
    # 选择几个代表性年份
    sample_years = [1896, 1936, 1976, 2000, 2024]

    plt.figure(figsize=(15, 10))
    for i, year in enumerate(sample_years, 1):
        year_data = df[df['Year'] == year].sort_values('Rank')
        plt.subplot(2, 3, i)
        plt.plot(year_data['Rank'], year_data['Total'], 'o-')
        plt.title(f'Year {year}')
        plt.xlabel('Rank')
        plt.ylabel('Total Medals')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rank_medal_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 计算排名与奖牌数的相关性
    correlations = df.groupby('Year').apply(
        lambda x: stats.spearmanr(x['Rank'], x['Total'])[0]
    ).round(3)

    print("\n3. Rank-Medal Correlation Analysis:")
    print("\nSpearman Correlation by Period:")
    correlations = pd.DataFrame(correlations)
    correlations.columns = ['Correlation']
    correlations['Period'] = pd.cut(correlations.index,
                                    bins=[1890, 1912, 1952, 1988, 2024],
                                    labels=['1896-1912', '1920-1952',
                                            '1956-1988', '1992-2024'])
    print(correlations.groupby('Period')['Correlation'].mean())


# 4. 集团差距分析
def analyze_group_gaps():
    def calculate_group_stats(year_data):
        # 定义第一集团（前8名）和第二集团（9-16名）
        group1 = year_data.head(8)
        group2 = year_data.iloc[8:16]

        return pd.Series({
            'Group1_Avg': group1['Total'].mean(),
            'Group2_Avg': group2['Total'].mean(),
            'Gap': group1['Total'].mean() - group2['Total'].mean(),
            'Gap_Ratio': group1['Total'].mean() / group2['Total'].mean()
        })

    group_analysis = df.groupby('Year').apply(calculate_group_stats)

    print("\n4. Group Gap Analysis:")
    print("\nAverage Gap by Period:")
    group_analysis['Period'] = pd.cut(group_analysis.index,
                                      bins=[1890, 1912, 1952, 1988, 2024],
                                      labels=['1896-1912', '1920-1952',
                                              '1956-1988', '1992-2024'])
    print(group_analysis.groupby('Period')[['Gap', 'Gap_Ratio']].mean().round(2))

    # 可视化集团差距趋势
    plt.figure(figsize=(15, 6))
    plt.plot(group_analysis.index, group_analysis['Gap_Ratio'], 'o-')
    plt.title('Gap Ratio Between Top 8 and Next 8 Countries')
    plt.xlabel('Year')
    plt.ylabel('Gap Ratio (Top 8 Avg / Next 8 Avg)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(group_analysis.index, rotation=45)
    plt.tight_layout()
    plt.savefig('group_gap_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

# 执行所有分析
analyze_medal_concentration()
analyze_rank_medal_relationship()
analyze_group_gaps()