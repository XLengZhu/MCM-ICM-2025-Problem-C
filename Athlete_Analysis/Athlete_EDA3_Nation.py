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
df = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


def noc_dimension_analysis():
    print("\n1. NOC/Region Dimension Analysis")

    # 1. 基础参与分析
    print("\n1.1 Basic Participation Analysis:")

    # 计算每个NOC的基本统计
    noc_stats = df.groupby('NOC').agg({
        'Name': 'count',  # 总参与次数
        'Year': 'nunique',  # 参与届数
        'Event': 'nunique',  # 参与具体项目数
        'Sport': 'nunique',  # 参与大项数
        'Medal': lambda x: (x != 'No medal').sum() # 奖牌数
    }).reset_index()

    noc_stats.columns = ['NOC', 'Participations', 'Olympics_Attended',
                         'Events_Participated', 'Sports_Participated', 'Total_Medals']

    # 计算人均奖牌数和项目覆盖率
    noc_stats['Medals_per_Participation'] = (noc_stats['Total_Medals'] /
                                             noc_stats['Participations']).round(3)
    noc_stats['Events_per_Sport'] = (noc_stats['Events_Participated'] /
                                     noc_stats['Sports_Participated']).round(2)

    print("\nTop 10 NOCs by total participation:")
    print(noc_stats.nlargest(10, 'Participations')[
              ['NOC', 'Participations', 'Olympics_Attended', 'Events_Participated',
               'Sports_Participated', 'Events_per_Sport', 'Total_Medals']])

    # 2. 参与持续性分析
    print("\n1.2 Participation Continuity Analysis:")
    noc_years = df.groupby(['NOC', 'Year']).size().reset_index()
    noc_years_pivot = noc_years.pivot(index='NOC', columns='Year', values=0)

    # 计算连续参与情况
    continuous_participation = noc_years_pivot.notna().astype(int)
    max_continuous = continuous_participation.sum(axis=1)

    print("\nNOCs with most Olympic appearances:")
    print(max_continuous.nlargest(10))

    # 3. 项目优势分析
    print("\n1.3 Event Advantage Analysis:")

    # 计算每个NOC在各个具体项目上的奖牌数
    event_medals = df[df['Medal']!="No medal"].groupby(['NOC', 'Sport', 'Event']).size()
    event_medals = event_medals.reset_index(name='Medal_Count')

    # 找出每个NOC最具优势的具体项目
    top_events = event_medals.sort_values('Medal_Count', ascending=False).groupby('NOC').first()

    print("\nStrength events for top 10 NOCs (by total medals):")
    top_nocs = noc_stats.nlargest(10, 'Total_Medals')['NOC']
    print(top_events.loc[top_nocs])

    # 4. 性别分布分析
    print("\n1.4 Gender Distribution Analysis:")
    gender_by_noc = pd.crosstab(df['NOC'], df['Sex'])
    gender_by_noc['F_Ratio'] = (gender_by_noc['F'] /
                                (gender_by_noc['F'] + gender_by_noc['M'])).round(3)

    print("\nNOCs with highest female participation ratio (min 100 participants):")
    min_participants = 100
    qualified_nocs = gender_by_noc[gender_by_noc.sum(axis=1) >= min_participants]
    print(qualified_nocs.nlargest(10, 'F_Ratio')[['F', 'M', 'F_Ratio']])

    # 5. 项目分布分析
    print("\n1.5 Event Distribution Analysis:")
    event_distribution = df.groupby('NOC').agg({
        'Event': ['nunique', 'count']
    }).reset_index()
    event_distribution.columns = ['NOC', 'Unique_Events', 'Total_Participations']
    event_distribution['Events_Coverage'] = (event_distribution['Unique_Events'] /
                                             df['Event'].nunique() * 100).round(2)

    print("\nTop 10 NOCs by event coverage (%):")
    print(event_distribution.nlargest(10, 'Events_Coverage'))

    # 6. 奖牌效率分析
    print("\n1.6 Medal Efficiency Analysis:")

    # 计算奖牌效率指标（考虑参与项目数）
    noc_stats['Medals_per_Event'] = (noc_stats['Total_Medals'] /
                                     noc_stats['Events_Participated']).round(2)

    print("\nTop 10 NOCs by medal efficiency (min 5 Olympics):")
    min_olympics = 5
    efficient_nocs = noc_stats[noc_stats['Olympics_Attended'] >= min_olympics]
    print(efficient_nocs.nlargest(10, 'Medals_per_Event')[
              ['NOC', 'Total_Medals', 'Events_Participated', 'Medals_per_Event']])


# 运行分析
noc_dimension_analysis()