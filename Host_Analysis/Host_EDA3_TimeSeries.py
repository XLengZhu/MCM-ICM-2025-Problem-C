import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_hosts.csv', encoding='utf-8')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

def temporal_evolution_analysis():
    print("\n1. Temporal Evolution Analysis")

    # 清理数据
    df['Country'] = df['Host'].apply(lambda x: x.split(',')[-1].strip())
    df['City'] = df['Host'].apply(lambda x: ','.join(x.split(',')[:-1]).strip())

    # 1. 主办权分配的时间线
    print("\n1.1 Timeline of Olympic Hosting:")
    timeline = df.groupby('Year').agg({
        'Country': 'first',
        'Host': 'first'
    }).reset_index()
    print(timeline)

    # 2. 分析每25年的变化
    bins = [1895, 1920, 1945, 1970, 1995, 2020, 2045]
    labels = ['1896-1920', '1921-1945', '1946-1970', '1971-1995', '1996-2020', '2021-2045']
    df['Period'] = pd.cut(df['Year'], bins=bins, labels=labels)

    period_stats = df.groupby('Period').agg({
        'Country': ['count', 'nunique'],
        'City': 'nunique'
    })
    period_stats.columns = ['Total_Games', 'Unique_Countries', 'Unique_Cities']

    print("\n1.2 Changes by 25-Year Periods:")
    print(period_stats)

    # 可视化每25年的变化
    plt.figure(figsize=(12, 6))
    period_stats.plot(kind='bar', width=0.8)
    plt.title('Olympic Games Distribution by 25-Year Periods')
    plt.xlabel('Period')
    plt.ylabel('Count')
    plt.legend(title='Metric')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('period_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 分析主办间隔的演变
    df['Host_Interval'] = df['Year'].diff()

    print("\n1.3 Hosting Intervals Analysis:")
    print("\nInterval Statistics by Period:")
    interval_stats = df.groupby('Period')['Host_Interval'].agg(['mean', 'min', 'max']).round(2)
    print(interval_stats)

    # 可视化主办间隔的变化
    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['Host_Interval'], marker='o')
    plt.title('Evolution of Hosting Intervals')
    plt.xlabel('Year')
    plt.ylabel('Interval (Years)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('hosting_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 重复主办国分析
    multiple_hosts = df['Country'].value_counts()
    years_between_hosts = {}

    for country in multiple_hosts[multiple_hosts > 1].index:
        country_years = sorted(df[df['Country'] == country]['Year'].values)
        intervals = np.diff(country_years)
        years_between_hosts[country] = {
            'avg_interval': np.mean(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals),
            'times_hosted': len(country_years)
        }

    print("\n1.4 Multiple Hosting Analysis:")
    for country, stats in years_between_hosts.items():
        print(f"\n{country}:")
        print(f"Times hosted: {stats['times_hosted']}")
        print(f"Average interval: {stats['avg_interval']:.1f} years")
        print(f"Range: {stats['min_interval']} to {stats['max_interval']} years")

    # 5. 特殊时期分析
    special_periods = {
        'Pre-WWI': (1896, 1912),
        'Interwar': (1920, 1936),
        'Post-WWII': (1948, 1968),
        'Cold War': (1968, 1988),
        'Modern Era': (1992, 2024)
    }

    print("\n1.5 Special Periods Analysis:")
    for period, (start, end) in special_periods.items():
        period_data = df[(df['Year'] >= start) & (df['Year'] <= end)]
        print(f"\n{period} ({start}-{end}):")
        print(f"Number of Games: {len(period_data)}")
        print(f"Unique Countries: {period_data['Country'].nunique()}")
        print(f"Countries: {', '.join(period_data['Country'].unique())}")


# 运行分析
temporal_evolution_analysis()