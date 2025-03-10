import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_hosts.csv', encoding='utf-8')


def basic_statistics():
    print("\n1. Basic Statistics")

    # 基本信息
    print("\n1.1 Dataset Info:")
    print(f"Number of Records: {len(df)}")
    print(f"Time Range: {df['Year'].min()} - {df['Year'].max()}")
    print("\nRaw data first few rows:")
    print(df.head())

    # 首先检查Host字段的格式
    print("\nUnique Host values:")
    print(df['Host'].unique())

    # 提取国家（假设国家总是在最后一个逗号之后）
    df['Country'] = df['Host'].apply(lambda x: x.split(',')[-1].strip())
    df['City'] = df['Host'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())

    # 统计主办次数
    country_counts = df['Country'].value_counts()
    city_counts = df['City'].value_counts()

    print("\n1.2 Host Countries Statistics:")
    print(f"Total number of unique host countries: {len(country_counts)}")
    print("\nCountries hosting multiple times:")
    print(country_counts[country_counts > 1])

    print("\n1.3 Host Cities Statistics:")
    print(f"Total number of unique host cities: {len(city_counts)}")
    print("\nCities hosting multiple times:")
    print(city_counts[city_counts > 1])

    # 可视化主办国分布
    plt.figure(figsize=(12, 6))
    country_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Olympic Host Countries')
    plt.xlabel('Country')
    plt.ylabel('Number of Times Hosted')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('host_countries_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 分析主办间隔
    df['Year_Diff'] = df['Year'].diff()

    print("\n1.4 Time Interval Statistics:")
    print("\nInterval between Olympic Games:")
    print(df['Year_Diff'].describe().round(2))

    # 可视化举办间隔的分布
    plt.figure(figsize=(10, 5))
    df['Year_Diff'].hist(bins=15)
    plt.title('Distribution of Time Intervals Between Olympics')
    plt.xlabel('Years Between Games')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_intervals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 分析每个时期的主办特点
    df['Period'] = pd.cut(df['Year'],
                          bins=[1890, 1912, 1952, 1988, 2032],
                          labels=['1896-1912', '1920-1952',
                                  '1956-1988', '1992-2032'])

    period_stats = df.groupby('Period').agg({
        'Country': ['count', 'nunique'],
        'City': 'nunique'
    })
    period_stats.columns = ['Total_Games', 'Unique_Countries', 'Unique_Cities']

    print("\n1.5 Statistics by Historical Period:")
    print(period_stats)

    # 可视化不同时期的主办国家数量
    plt.figure(figsize=(12, 6))
    period_stats[['Unique_Countries', 'Unique_Cities']].plot(kind='bar')
    plt.title('Number of Host Countries and Cities by Period')
    plt.xlabel('Period')
    plt.ylabel('Count')
    plt.legend(title='Type')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('hosts_by_period.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 分析连续主办的情况
    df['Same_Country_Next'] = (df['Country'] == df['Country'].shift(-1))
    consecutive_hosts = df[df['Same_Country_Next']][['Year', 'Country', 'City']]

    print("\n1.6 Consecutive Hosting Analysis:")
    print("\nCountries hosting in consecutive games:")
    print(consecutive_hosts)


# 运行分析
basic_statistics()