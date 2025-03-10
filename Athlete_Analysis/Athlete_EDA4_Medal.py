import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


def medal_dimension_analysis():
    print("\n1. Medal Dimension Analysis")

    # 1. 基础奖牌分布
    print("\n1.1 Basic Medal Distribution:")
    medal_dist = df['Medal'].value_counts()
    print("\nOverall Medal Distribution:")
    print(medal_dist)

    # 计算获奖率
    medal_rate = (len(df[df['Medal']!="No medal"])/ len(df) * 100)
    print(f"\nOverall Medal Rate: {medal_rate}%")

    # 2. 奖牌的时间趋势
    print("\n1.2 Medal Distribution Over Time:")
    medals_by_year = df[df['Medal']!="No medal"].groupby(['Year', 'Medal']).size().unstack()

    # 计算每届奖牌比例
    medals_ratio = medals_by_year.div(medals_by_year.sum(axis=1), axis=0) * 100
    print("\nMedal Ratio by Year (%):")
    print(medals_ratio.round(2))

    # 绘制奖牌数量趋势
    plt.figure(figsize=(12, 6))
    medals_by_year.plot(kind='bar', stacked=True)
    plt.title('Number of Medals by Type Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Medals')
    plt.legend(title='Medal Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('medals_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 性别与奖牌分析
    print("\n1.3 Gender and Medal Analysis:")
    gender_medals = pd.crosstab(df['Sex'], df['Medal'])
    gender_medal_rates = (gender_medals.div(gender_medals.sum(axis=1), axis=0) * 100).round(2)
    print("\nMedal Distribution by Gender (%):")
    print(gender_medal_rates)

    # 4. 项目奖牌分布
    print("\n1.4 Medal Distribution by Event:")
    event_medals = df[df['Medal']!="No medal"].groupby(['Sport', 'Event']).size().reset_index(name='Medal_Count')
    event_medals = event_medals.sort_values('Medal_Count', ascending=False)

    print("\nTop 10 Events by Number of Medals:")
    print(event_medals.head(10))

    # 5. 国家奖牌集中度
    print("\n1.5 Medal Concentration Analysis:")
    noc_medals = df[df['Medal']!="No medal"].groupby('NOC').size().sort_values(ascending=False)
    top_nocs_share = (noc_medals.head(10).sum() / noc_medals.sum() * 100).round(2)
    print(f"\nTop 10 NOCs Share of All Medals: {top_nocs_share}%")

    # 计算基尼系数
    def gini(x):
        x = np.array(x)
        n = len(x)
        s = x.sum()
        r = np.argsort(np.argsort(-x))  # rank in descending order
        return (2 * (r * x).sum() / (n * s) - (n + 1) / n).round(3)

    gini_coef = gini(noc_medals.values)
    print(f"\nGini Coefficient for Medal Distribution: {gini_coef}")

    # 6. 奖牌转换率分析
    print("\n1.6 Medal Conversion Rate Analysis:")
    # 按NOC和项目计算参与次数和获奖次数
    participation = df.groupby(['NOC', 'Event']).size().reset_index(name='Participations')
    medals = df[df['Medal']!="No medal"].groupby(['NOC', 'Event']).size().reset_index(name='Medals')

    # 合并数据
    conversion = pd.merge(participation, medals, on=['NOC', 'Event'], how='left')
    conversion['Conversion_Rate'] = (conversion['Medals'] / conversion['Participations'] * 100).round(2)

    # 计算各NOC的平均转换率
    noc_conversion = conversion.groupby('NOC')['Conversion_Rate'].mean().round(2)

    print("\nTop 10 NOCs by Medal Conversion Rate (minimum 100 participations):")
    qualified_conversion = noc_conversion[participation.groupby('NOC')['Participations'].sum() >= 100]
    print(qualified_conversion.nlargest(10))


# 运行分析
medal_dimension_analysis()