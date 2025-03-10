import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


def basic_statistics_analysis():
    print("\n1. Basic Statistics Analysis")

    # 1.1 基本数据信息
    print("\n1.1 Dataset Basic Information:")
    print(f"Total Records: {len(df)}")
    print("\nColumns:", list(df.columns))
    print("\nData Types:")
    print(df.dtypes)

    # 1.2 缺失值分析
    print("\n1.2 Missing Value Analysis:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")

    # 1.3 时间范围分析
    print("\n1.3 Time Range Analysis:")
    print(f"Years covered: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Number of Olympics: {df['Year'].nunique()}")

    yearly_stats = df.groupby('Year').agg({
        'Name': 'count',
        'NOC': 'nunique',
        'Sport': 'nunique'
    }).reset_index()

    print("\nParticipation Statistics by Year (first 5 and last 5 years):")
    print("\nFirst 5 Olympics:")
    print(yearly_stats.head())
    print("\nLast 5 Olympics:")
    print(yearly_stats.tail())

    # 可视化参赛人数的历史变化
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_stats['Year'], yearly_stats['Name'], marker='o')
    plt.title('Number of Olympic Participants Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Participants')
    plt.grid(True, alpha=0.3)
    plt.xticks(yearly_stats['Year'],rotation=45)
    plt.tight_layout()
    plt.savefig('participants_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 1.4 性别分布分析
    print("\n1.4 Gender Distribution Analysis:")
    gender_dist = df['Sex'].value_counts()
    print("\nOverall gender distribution:")
    print(gender_dist)

    # 计算性别比例
    gender_ratio = gender_dist['M'] / gender_dist['F']
    print(f"\nGender Ratio (M/F): {gender_ratio:.2f}")

    # 计算每届奥运会的性别比例
    gender_by_year = pd.crosstab(df['Year'], df['Sex'])
    gender_ratio_by_year = gender_by_year['M'] / gender_by_year['F']

    plt.figure(figsize=(12, 6))
    plt.plot(gender_ratio_by_year.index, gender_ratio_by_year.values, marker='o')
    plt.title('Gender Ratio (M/F) Over Time')
    plt.xlabel('Year')
    plt.ylabel('Ratio (M/F)')
    plt.grid(True, alpha=0.3)
    plt.xticks(gender_ratio_by_year.index,rotation=45)
    plt.tight_layout()
    plt.savefig('gender_ratio_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 1.5 NOC和Team分析
    print("\n1.5 NOC and Team Analysis:")
    print(f"Number of unique NOCs: {df['NOC'].nunique()}")
    print(f"Number of unique Teams: {df['Team'].nunique()}")

    top_nocs = df['NOC'].value_counts().head(10)
    print("\nTop 10 NOCs by participation:")
    print(top_nocs)

    # 1.6 运动项目分析
    print("\n1.6 Sports Analysis:")
    print(f"Number of unique Sports: {df['Sport'].nunique()}")
    print(f"Number of unique Events: {df['Event'].nunique()}")

    top_sports = df['Sport'].value_counts().head(10)
    print("\nTop 10 Sports by participation:")
    print(top_sports)

    # 1.7 奖牌分析
    print("\n1.7 Medal Analysis:")
    medal_dist = df['Medal'].value_counts(dropna=False)
    print("\nMedal Distribution:")
    print(medal_dist)

    # 计算获奖率（排除"No medal"）
    medals_won = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])].shape[0]
    medal_rate = (medals_won / len(df)) * 100
    print(f"\nMedal winning rate: {medal_rate:.2f}%")

    # 1.8 城市分析
    print("\n1.8 Host Cities Analysis:")
    print(f"Number of unique host cities: {df['City'].nunique()}")

    top_cities = df.groupby('City').size().sort_values(ascending=False).head(10)
    print("\nTop 10 host cities by number of participants:")
    print(top_cities)

    # 1.9 参与者分析
    print("\n1.9 Participant Analysis:")
    # 使用Name和NOC的组合作为唯一标识
    df['Athlete_ID'] = df['Name'] + '_' + df['NOC']

    # 统计基本参与信息
    unique_athletes = df['Athlete_ID'].nunique()
    print(f"Number of unique athletes: {unique_athletes}")

    # 计算参赛次数
    appearances = df.groupby('Athlete_ID').size()
    multiple_appearances = appearances[appearances > 1]

    print(f"\nAthletes with multiple appearances: {len(multiple_appearances)}")
    print(f"Maximum appearances by an athlete: {appearances.max()}")

    # 显示参赛次数最多的前10名运动员
    top_appearances = appearances.nlargest(10)
    print("\nTop 10 athletes by number of appearances:")
    for athlete_id, count in top_appearances.items():
        name, noc = athlete_id.rsplit('_', 1)
        print(f"{name} ({noc}): {count} appearances")


# 运行分析
basic_statistics_analysis()