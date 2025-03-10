import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')


# 计算每届奥运会的奖牌集中度
def calculate_concentration(year_data):
    # 计算前3名、前10名的奖牌占比
    total_medals = year_data['Total'].sum()
    top3_share = year_data.nlargest(3, 'Total')['Total'].sum() / total_medals * 100
    top10_share = year_data.nlargest(10, 'Total')['Total'].sum() / total_medals * 100

    return pd.Series({
        'Top3_Share': top3_share,
        'Top10_Share': top10_share,
        'Total_Countries': len(year_data),
        'Countries_with_Medals': len(year_data[year_data['Total'] > 0])
    })


concentration_stats = df.groupby('Year').apply(calculate_concentration).reset_index()

# 可视化奖牌集中度趋势
plt.figure(figsize=(15, 7))
plt.plot(concentration_stats['Year'], concentration_stats['Top3_Share'], 'o-',
         label='Top 3 Countries', color='darkred')
plt.plot(concentration_stats['Year'], concentration_stats['Top10_Share'], 's-',
         label='Top 10 Countries', color='navy')

plt.title('Olympic Medal Concentration Over Time')
plt.xlabel('Year')
plt.ylabel('Share of Total Medals (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(concentration_stats['Year'], rotation=45)
plt.tight_layout()
plt.savefig('medal_concentration.png', dpi=300, bbox_inches='tight')
plt.close()

# 输出统计信息
print("\nMedal Concentration Analysis:")
print("\nAverage concentration by period:")
concentration_stats['Period'] = pd.cut(concentration_stats['Year'],
                                       bins=[1890, 1912, 1952, 1988, 2024],
                                       labels=['1896-1912', '1920-1952',
                                               '1956-1988', '1992-2024'])

period_avg = concentration_stats.groupby('Period').agg({
    'Top3_Share': 'mean',
    'Top10_Share': 'mean',
    'Countries_with_Medals': 'mean'
}).round(2)

print(period_avg)

print("\nYears with highest concentration (Top 3 countries):")
print(concentration_stats.nlargest(5, 'Top3_Share')[
          ['Year', 'Top3_Share', 'Countries_with_Medals']])

print("\nYears with lowest concentration (Top 3 countries):")
print(concentration_stats.nsmallest(5, 'Top3_Share')[
          ['Year', 'Top3_Share', 'Countries_with_Medals']])