import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_hosts.csv', encoding='utf-8')


def geographic_analysis():
    print("\n1. Geographic Distribution Analysis")

    # 清理数据并添加洲际分类
    df['Country'] = df['Host'].apply(lambda x: x.split(',')[-1].strip())

    # 手动添加洲际分类
    continent_map = {
        'United States': 'North America',
        'France': 'Europe',
        'Greece': 'Europe',
        'United Kingdom': 'Europe',
        'Sweden': 'Europe',
        'Belgium': 'Europe',
        'Netherlands': 'Europe',
        'Germany': 'Europe',
        'Finland': 'Europe',
        'Italy': 'Europe',
        'Australia': 'Oceania',
        'Japan': 'Asia',
        'Mexico': 'North America',
        'West Germany': 'Europe',
        'Canada': 'North America',
        'Soviet Union': 'Europe',
        'South Korea': 'Asia',
        'Spain': 'Europe',
        'China': 'Asia',
        'Brazil': 'South America'
    }

    df['Continent'] = df['Country'].map(continent_map)

    # 1. 洲际分布统计
    continent_stats = df.groupby('Continent')['Year'].count().sort_values(ascending=False)

    print("\n1.1 Continental Distribution:")
    print(continent_stats)

    # 可视化洲际分布
    plt.figure(figsize=(12, 6))
    continent_stats.plot(kind='bar')
    plt.title('Distribution of Olympic Games by Continent')
    plt.xlabel('Continent')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('continental_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 时期分析
    df['Period'] = pd.cut(df['Year'],
                          bins=[1890, 1912, 1952, 1988, 2032],
                          labels=['1896-1912', '1920-1952',
                                  '1956-1988', '1992-2032'])

    # 按时期统计洲际分布
    period_continent = pd.crosstab(df['Period'], df['Continent'])

    print("\n1.2 Continental Distribution by Period:")
    print(period_continent)

    # 可视化时期-洲际分布
    plt.figure(figsize=(12, 6))
    period_continent.plot(kind='bar', stacked=True)
    plt.title('Continental Distribution of Olympic Games by Period')
    plt.xlabel('Period')
    plt.ylabel('Number of Games')
    plt.legend(title='Continent', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('continental_distribution_by_period.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 国家重复主办分析
    country_freq = df['Country'].value_counts()

    print("\n1.3 Countries Hosting Multiple Times:")
    print(country_freq[country_freq > 1])

    # 计算每个洲的平均间隔
    def calculate_continent_intervals(continent_data):
        if len(continent_data) <= 1:
            return np.nan
        intervals = continent_data['Year'].sort_values().diff().mean()
        return intervals

    continent_intervals = df.groupby('Continent').apply(calculate_continent_intervals)

    print("\n1.4 Average Interval Between Games by Continent (years):")
    print(continent_intervals)

    # 4. 分析新主办国的地理分布
    df_sorted = df.sort_values('Year')
    df_sorted['First_Time'] = ~df_sorted['Country'].duplicated()

    first_time_hosts = df_sorted[df_sorted['First_Time']]
    first_time_by_continent = pd.crosstab(first_time_hosts['Period'],
                                          first_time_hosts['Continent'])

    print("\n1.5 First-Time Hosts by Period and Continent:")
    print(first_time_by_continent)


# 运行分析
geographic_analysis()