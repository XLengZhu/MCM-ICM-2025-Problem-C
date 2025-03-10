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
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')

# 1. 历史奖牌榜分析
historical_medals = df.groupby('NOC').agg({
    'Gold': 'sum',
    'Silver': 'sum',
    'Bronze': 'sum',
    'Total': 'sum',
    'Year': 'count'  # 参与届数
}).reset_index()

historical_medals['Avg_Medals_per_Game'] = (historical_medals['Total'] /
                                          historical_medals['Year']).round(2)
historical_medals['Gold_Ratio'] = (historical_medals['Gold'] /
                                 historical_medals['Total'] * 100).round(2)

# 输出前20名的详细信息
print("\nTop 20 Countries in Olympic History:")
print(historical_medals.nlargest(20, 'Total')[
    ['NOC', 'Gold', 'Silver', 'Bronze', 'Total', 'Year',
     'Avg_Medals_per_Game', 'Gold_Ratio']])

# 可视化前10名国家的奖牌构成
plt.figure(figsize=(15, 8))
top10_countries = historical_medals.nlargest(10, 'Total')

bar_width = 0.25
index = np.arange(len(top10_countries))

plt.bar(index, top10_countries['Gold'], bar_width, label='Gold', color='gold')
plt.bar(index, top10_countries['Silver'], bar_width,
        bottom=top10_countries['Gold'], label='Silver', color='silver')
plt.bar(index, top10_countries['Bronze'], bar_width,
        bottom=top10_countries['Gold'] + top10_countries['Silver'],
        label='Bronze', color='brown')

plt.title('Medal Distribution of Top 10 Countries in Olympic History')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.xticks(index, top10_countries['NOC'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('top10_countries_medals.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 奖牌效率分析
# 计算每个国家在其参加的奥运会中的平均表现
medal_efficiency = df.groupby('NOC').agg({
    'Total': ['mean', 'max', 'min', 'std'],
    'Year': 'count'
}).round(2)

medal_efficiency.columns = ['Avg_Medals', 'Max_Medals', 'Min_Medals',
                          'Std_Medals', 'Participations']
medal_efficiency = medal_efficiency.reset_index()

# 筛选参与至少10届的国家
regular_participants = medal_efficiency[medal_efficiency['Participations'] >= 10]
print("\nMedal Efficiency for Countries with 10+ Participations:")
print(regular_participants.nlargest(15, 'Avg_Medals'))

# 可视化奖牌效率
plt.figure(figsize=(15, 8))
top15_efficient = regular_participants.nlargest(15, 'Avg_Medals')

plt.bar(np.arange(len(top15_efficient)), top15_efficient['Avg_Medals'],
        color='skyblue')
plt.errorbar(np.arange(len(top15_efficient)), top15_efficient['Avg_Medals'],
             yerr=top15_efficient['Std_Medals'], fmt='none', color='black',
             capsize=5)

plt.title('Medal Efficiency of Top 15 Countries\n(Minimum 10 Participations)')
plt.xlabel('Country')
plt.ylabel('Average Medals per Olympics')
plt.xticks(np.arange(len(top15_efficient)), top15_efficient['NOC'], rotation=45)
plt.tight_layout()
plt.savefig('medal_efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 分析平均金牌数
gold_efficiency = df[df['Gold'] > 0].groupby('NOC').agg({
    'Gold': ['mean', 'sum'],
    'Year': 'count'
}).round(2)

gold_efficiency.columns = ['Avg_Gold', 'Total_Gold',
                         'Total_Games']
print("\nGold Medal Efficiency (Countries with at least 5 gold medals):")
print(gold_efficiency[gold_efficiency['Total_Gold'] >= 5].sort_values(
    'Avg_Gold', ascending=False).head(15))