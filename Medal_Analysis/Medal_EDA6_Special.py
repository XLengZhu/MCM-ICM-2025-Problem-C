import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')


# 1. 首次获得奖牌的国家分析
def analyze_first_medals():
    print("\n1. First Medal Analysis")

    # 按国家排序并找出首次获奖情况
    first_medals = df[df['Total'] > 0].sort_values('Year').groupby('NOC').first().reset_index()

    # 按时期统计首次获奖国家数
    first_medals['Period'] = pd.cut(first_medals['Year'],
                                    bins=[1890, 1912, 1952, 1988, 2024],
                                    labels=['1896-1912', '1920-1952',
                                            '1956-1988', '1992-2024'])

    period_stats = first_medals['Period'].value_counts().sort_index()
    print("\nNumber of countries winning their first medal by period:")
    print(period_stats)

    # 最近20年首次获奖的国家
    recent_first = first_medals[first_medals['Year'] >= 2004]
    print("\nCountries winning their first medal since 2004:")
    print(recent_first[['NOC', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']])


# 2. 突破性增长分析
def analyze_breakthrough_performances():
    print("\n2. Breakthrough Performance Analysis")

    # 计算奖牌数的年度变化
    df_sorted = df.sort_values(['NOC', 'Year'])
    df_sorted['Prev_Total'] = df_sorted.groupby('NOC')['Total'].shift(1)
    df_sorted['Growth'] = df_sorted['Total'] - df_sorted['Prev_Total']
    df_sorted['Growth_Rate'] = (df_sorted['Growth'] / df_sorted['Prev_Total'] * 100).round(2)

    # 筛选显著增长案例 (增长率>100%且至少增加5枚奖牌)
    significant_growth = df_sorted[
        (df_sorted['Growth_Rate'].notna()) &
        (df_sorted['Growth_Rate'] > 100) &
        (df_sorted['Growth'] >= 5)
        ].sort_values('Growth_Rate', ascending=False)

    print("\nMost significant breakthrough cases:")
    print(significant_growth[['Year', 'NOC', 'Total', 'Prev_Total', 'Growth', 'Growth_Rate']].head(15))

    # 分析持续增长的国家
    df_sorted['Next_Total'] = df_sorted.groupby('NOC')['Total'].shift(-1)
    df_sorted['Next2_Total'] = df_sorted.groupby('NOC')['Total'].shift(-2)

    # 找出连续三届增长的情况
    continuous_growth = df_sorted[
        (df_sorted['Total'] < df_sorted['Next_Total']) &
        (df_sorted['Next_Total'] < df_sorted['Next2_Total'])
        ]

    growth_counts = continuous_growth.groupby('NOC').size().sort_values(ascending=False)
    print("\nCountries with most periods of continuous growth:")
    print(growth_counts.head(10))


# 3. 排名波动分析
def analyze_ranking_volatility():
    print("\n3. Ranking Volatility Analysis")

    # 计算排名变化
    df_sorted = df.sort_values(['NOC', 'Year'])
    df_sorted['Prev_Rank'] = df_sorted.groupby('NOC')['Rank'].shift(1)
    df_sorted['Rank_Change'] = df_sorted['Rank'] - df_sorted['Prev_Rank']

    # 找出最大排名提升（负值表示排名提升）
    improvements = df_sorted[df_sorted['Rank_Change'].notna()]
    biggest_improvements = improvements.nlargest(10, 'Rank_Change')

    print("\nBiggest ranking improvements in a single Olympics:")
    print(biggest_improvements[['Year', 'NOC', 'Prev_Rank', 'Rank', 'Rank_Change', 'Total']])

    # 找出最大排名下降
    biggest_drops = improvements.nsmallest(10, 'Rank_Change')
    print("\nBiggest ranking drops in a single Olympics:")
    print(biggest_drops[['Year', 'NOC', 'Prev_Rank', 'Rank', 'Rank_Change', 'Total']])

    # 分析长期趋势
    long_term = df_sorted.groupby('NOC').agg({
        'Rank': ['first', 'last', 'count']
    }).reset_index()

    long_term.columns = ['NOC', 'Initial_Rank', 'Final_Rank', 'Appearances']
    long_term['Rank_Change'] = long_term['Final_Rank'] - long_term['Initial_Rank']

    # 筛选至少参加5届的国家
    long_term = long_term[long_term['Appearances'] >= 5]

    print("\nMost improved countries over their Olympic history (min 5 appearances):")
    print(long_term.nsmallest(10, 'Rank_Change')[
              ['NOC', 'Initial_Rank', 'Final_Rank', 'Rank_Change', 'Appearances']])

    print("\nMost declined countries over their Olympic history (min 5 appearances):")
    print(long_term.nlargest(10, 'Rank_Change')[
              ['NOC', 'Initial_Rank', 'Final_Rank', 'Rank_Change', 'Appearances']])


# 执行所有分析
analyze_first_medals()
analyze_breakthrough_performances()
analyze_ranking_volatility()