import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


def athlete_dimension_analysis():
    print("\n1. Athlete Dimension Analysis")

    # 创建运动员唯一标识
    df['Athlete_ID'] = df['Name'] + '_' + df['NOC']

    # 1. 运动员参赛经历分析
    print("\n1.1 Olympic Career Analysis:")

    # 计算每个运动员的参赛经历
    athlete_careers = df.groupby('Athlete_ID').agg({
        'Year': ['count', 'min', 'max'],
        'Sport': 'nunique',
        'Medal': lambda x: (x != 'No medal').sum()
    }).reset_index()

    athlete_careers.columns = ['Athlete_ID', 'Appearances', 'First_Year', 'Last_Year',
                               'Sports_Count', 'Medals_Count']
    athlete_careers['Career_Span'] = athlete_careers['Last_Year'] - athlete_careers['First_Year']

    print("\nCareer Statistics:")
    print(athlete_careers[['Appearances', 'Career_Span', 'Sports_Count', 'Medals_Count']].describe())

    # 2. 运动员多项目参与分析
    print("\n1.2 Multi-Sport Athletes Analysis:")
    multi_sport = athlete_careers[athlete_careers['Sports_Count'] > 1]
    print(
        f"\nNumber of multi-sport athletes: {len(multi_sport)} ({len(multi_sport) / len(athlete_careers) * 100:.2f}%)")

    # 显示参与项目最多的运动员
    print("\nTop 10 athletes by number of different sports:")
    top_multi_sport = athlete_careers.nlargest(10, 'Sports_Count')
    for _, athlete in top_multi_sport.iterrows():
        name, noc = athlete['Athlete_ID'].split('_')
        print(f"{name} ({noc}): {athlete['Sports_Count']} sports over {athlete['Career_Span']} years")

    # 3. 奖牌获得者分析
    print("\n1.3 Medal Winners Analysis:")
    medalists = athlete_careers[athlete_careers['Medals_Count'] > 0]
    print(f"\nNumber of medal winners: {len(medalists)} ({len(medalists) / len(athlete_careers) * 100:.2f}%)")

    # 显示获得奖牌最多的运动员
    print("\nTop 10 athletes by number of medals:")
    top_medalists = medalists.nlargest(10, 'Medals_Count')
    for _, athlete in top_medalists.iterrows():
        name, noc = athlete['Athlete_ID'].split('_')
        print(f"{name} ({noc}): {athlete['Medals_Count']} medals")

    # 4. 职业生涯持续时间分析
    plt.figure(figsize=(12, 6))
    athlete_careers['Career_Span'].hist(bins=50)
    plt.title('Distribution of Olympic Career Spans')
    plt.xlabel('Career Span (Years)')
    plt.ylabel('Number of Athletes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('career_spans.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 6. 性别相关分析
    print("\n1.4 Gender-based Analysis:")
    gender_stats = df.groupby('Sex').agg({
        'Athlete_ID': 'nunique',
        'Medal': lambda x: (x != 'No medal').sum()
    }).reset_index()

    gender_stats['Medal_per_Athlete'] = gender_stats['Medal'] / gender_stats['Athlete_ID']
    print("\nGender-based participation and performance:")
    print(gender_stats)

    # 7. 运动员国籍转换分析
    print("\n1.5 NOC Change Analysis:")
    athletes_with_multiple_nocs = df.groupby('Name')['NOC'].nunique()
    noc_changers = athletes_with_multiple_nocs[athletes_with_multiple_nocs > 1]

    print(f"\nAthletes who represented different NOCs: {len(noc_changers)}")
    if len(noc_changers) > 0:
        print("\nExample cases of NOC changes:")
        for name in noc_changers.head().index:
            noc_changes = df[df['Name'] == name][['Year', 'NOC']].sort_values('Year')
            print(f"\n{name}:")
            print(noc_changes)


# 运行分析
athlete_dimension_analysis()