import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 读取数据
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')


# 1. 金牌/总奖牌比率分析
def analyze_gold_ratio():
    print("\n1. Gold Medal Ratio Analysis")

    # 计算各国的金牌比率
    country_ratios = df[df['Total'] >= 10].groupby('NOC').agg({
        'Gold': 'sum',
        'Total': 'sum',
        'Year': 'count'
    }).assign(
        Gold_Ratio=lambda x: (x['Gold'] / x['Total'] * 100).round(2)
    ).sort_values('Gold_Ratio', ascending=False)

    print("\nTop 15 Countries by Gold Medal Ratio (minimum 10 total medals):")
    print(country_ratios.head(15)[['Gold', 'Total', 'Year', 'Gold_Ratio']])

    # 分析金牌比率的历史变化
    yearly_ratios = df.groupby('Year').agg({
        'Gold': 'sum',
        'Total': 'sum'
    }).assign(
        Gold_Ratio=lambda x: (x['Gold'] / x['Total'] * 100).round(2)
    )

    # 可视化金牌比率趋势
    plt.figure(figsize=(15, 6))
    plt.plot(yearly_ratios.index, yearly_ratios['Gold_Ratio'], 'o-', color='gold')
    plt.title('Gold Medal Ratio Trend Over Time')
    plt.xlabel('Year')
    plt.ylabel('Gold Medal Ratio (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_ratios.index,rotation=45)
    plt.tight_layout()
    plt.savefig('gold_ratio_trend.png', dpi=300, bbox_inches='tight')
    plt.close()


# 2. 排名稳定性分析
def analyze_ranking_stability():
    print("\n2. Ranking Stability Analysis")

    # 选择参加至少10届的国家
    frequent_countries = df.groupby('NOC').filter(lambda x: len(x) >= 10)

    # 计算排名的标准差和变异系数
    ranking_stability = frequent_countries.groupby('NOC').agg({
        'Rank': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)

    ranking_stability.columns = ['Avg_Rank', 'Std_Rank', 'Best_Rank', 'Worst_Rank', 'Participations']
    ranking_stability['Rank_CV'] = (ranking_stability['Std_Rank'] /
                                    ranking_stability['Avg_Rank'] * 100).round(2)

    print("\nRanking Stability of Major Countries:")
    print(ranking_stability.sort_values('Avg_Rank').head(15))

    # 计算排名跳跃
    def calculate_rank_jumps(country_data):
        rank_changes = country_data['Rank'].diff().abs()
        return pd.Series({
            'Avg_Rank_Change': rank_changes.mean(),
            'Max_Rank_Change': rank_changes.max()
        })

    rank_changes = frequent_countries.sort_values('Year').groupby('NOC').apply(calculate_rank_jumps)
    print("\nCountries with Most Volatile Rankings:")
    print(rank_changes.sort_values('Avg_Rank_Change', ascending=False).head(10))


# 3. TOP国家份额分析
def analyze_top_countries_share():
    print("\n3. Top Countries Share Analysis")

    def calculate_top_shares(year_data):
        total_medals = year_data['Total'].sum()
        total_gold = year_data['Gold'].sum()

        top3_medals = year_data.nlargest(3, 'Total')['Total'].sum()
        top10_medals = year_data.nlargest(10, 'Total')['Total'].sum()

        top3_gold = year_data.nlargest(3, 'Gold')['Gold'].sum()
        top10_gold = year_data.nlargest(10, 'Gold')['Gold'].sum()

        return pd.Series({
            'Top3_Medal_Share': (top3_medals / total_medals * 100).round(2),
            'Top10_Medal_Share': (top10_medals / total_medals * 100).round(2),
            'Top3_Gold_Share': (top3_gold / total_gold * 100).round(2),
            'Top10_Gold_Share': (top10_gold / total_gold * 100).round(2)
        })

    shares = df.groupby('Year').apply(calculate_top_shares)

    # 分时期分析
    shares['Period'] = pd.cut(shares.index,
                              bins=[1890, 1912, 1952, 1988, 2024],
                              labels=['1896-1912', '1920-1952',
                                      '1956-1988', '1992-2024'])

    print("\nAverage Shares by Period:")
    print(shares.groupby('Period').mean().round(2))

    # 可视化TOP国家份额趋势
    plt.figure(figsize=(15, 6))
    plt.plot(shares.index, shares['Top3_Medal_Share'], 'o-',
             label='Top 3 Share', color='blue')
    plt.plot(shares.index, shares['Top10_Medal_Share'], 's-',
             label='Top 10 Share', color='red')

    plt.title('Share of Medals by Top Countries Over Time')
    plt.xlabel('Year')
    plt.ylabel('Share of Total Medals (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_countries_share.png', dpi=300, bbox_inches='tight')
    plt.close()


# 执行所有分析
analyze_gold_ratio()
analyze_ranking_stability()
analyze_top_countries_share()