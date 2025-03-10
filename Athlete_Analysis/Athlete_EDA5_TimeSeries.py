import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_athletes.csv', encoding='utf-8')


def temporal_dimension_analysis():
    print("\n1. Temporal Dimension Analysis")

    # 1. 参与规模趋势
    print("\n1.1 Participation Scale Analysis:")
    yearly_stats = df.groupby('Year').agg({
        'Name': 'count',  # 总参与人数
        'NOC': 'nunique',  # 参与国家数
        'Event': 'nunique',  # 项目数
        'Sex': lambda x: (x == 'F').mean() * 100  # 女性比例
    }).round(2)

    yearly_stats.columns = ['Participants', 'NOCs', 'Events', 'Female_Ratio']
    print("\nParticipation trends (first and last 5 Olympics):")
    print("\nFirst 5 Olympics:")
    print(yearly_stats.head())
    print("\nLast 5 Olympics:")
    print(yearly_stats.tail())

    # 2. 新项目分析
    print("\n1.2 Event Evolution Analysis:")
    # 计算每届新增的项目
    events_by_year = df.groupby(['Year', 'Event']).size().reset_index()
    events_by_year = events_by_year.sort_values(['Event', 'Year'])
    events_by_year['First_Appearance'] = events_by_year.groupby('Event')['Year'].transform('min')
    new_events = events_by_year[events_by_year['Year'] == events_by_year['First_Appearance']]

    new_events_count = new_events.groupby('Year').size()
    print("\nNumber of new events introduced by year:")
    print(new_events_count)

    # 3. 竞争强度分析
    print("\n1.3 Competition Intensity Analysis:")
    # 计算每个项目的平均参与国家数
    competition_intensity = df.groupby(['Year', 'Event']).agg({
        'NOC': 'nunique'
    }).reset_index()

    yearly_intensity = competition_intensity.groupby('Year')['NOC'].mean().round(2)
    print("\nAverage number of participating countries per event:")
    print(yearly_intensity)

    # 4. 国家参与度变化
    print("\n1.4 NOC Participation Analysis:")
    # 计算每个NOC的参与年份范围
    noc_participation = df.groupby('NOC').agg({
        'Year': ['min', 'max', 'nunique']
    }).round(2)
    noc_participation.columns = ['First_Year', 'Last_Year', 'Olympics_Attended']
    noc_participation['Active_Span'] = noc_participation['Last_Year'] - noc_participation['First_Year']

    print("\nNOC participation patterns:")
    print(noc_participation.describe())

    # 5. 项目结构变化
    print("\n1.5 Event Structure Analysis:")
    # 分析不同类型项目的比例变化
    event_type_yearly = df.groupby(['Year', 'Sport']).agg({
        'Event': 'nunique',
        'Name': 'count'
    }).reset_index()

    # 计算每年各大项占比
    event_type_yearly['Event_Share'] = event_type_yearly.groupby('Year')['Event'].transform(
        lambda x: x / x.sum() * 100
    ).round(2)

    print("\nTop 5 sports by number of events (most recent Olympics):")
    recent_year = event_type_yearly['Year'].max()
    print(event_type_yearly[event_type_yearly['Year'] == recent_year].nlargest(5, 'Event')[
              ['Sport', 'Event', 'Event_Share']])

    # 6. 性别平等进展
    print("\n1.6 Gender Equality Progress:")
    gender_progress = df.groupby(['Year', 'Sex']).size().unstack(fill_value=0)
    gender_progress['F_Ratio'] = (gender_progress['F'] /
                                  (gender_progress['F'] + gender_progress['M']) * 100).round(2)

    print("\nFemale participation ratio over time:")
    print(gender_progress['F_Ratio'])

# 运行分析
temporal_dimension_analysis()