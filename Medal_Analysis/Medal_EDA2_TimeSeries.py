import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局绘图参数
plt.style.use('default')  # 使用默认样式
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['axes.grid'] = True

# 读取数据
df = pd.read_csv('../summerOly_medal_counts.csv', encoding='utf-8')

# 1. 每届奥运会的基本统计
yearly_stats = df.groupby('Year').agg({
    'NOC': 'count',  # Number of participating countries
    'Gold': 'sum',   # Total gold medals
    'Silver': 'sum', # Total silver medals
    'Bronze': 'sum', # Total bronze medals
    'Total': 'sum'   # Total medals
}).reset_index()

yearly_stats = yearly_stats.sort_values(by='Year').reset_index(drop=True)

print("\nBasic statistics for each Olympic Games:")
print(yearly_stats)

# Figure 1: Medal count trends
plt.figure(figsize=(15, 8))
plt.plot(yearly_stats['Year'], yearly_stats['Gold'], 'o-', label='Gold', color='gold')
plt.plot(yearly_stats['Year'], yearly_stats['Silver'], 's-', label='Silver', color='silver')
plt.plot(yearly_stats['Year'], yearly_stats['Bronze'], '^-', label='Bronze', color='brown')

plt.title('Olympic Games Medal Count Trends (1896-2024)', pad=20)
plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(yearly_stats['Year'], rotation=45)
plt.tight_layout()
plt.savefig('medals_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Participating countries trend
plt.figure(figsize=(15, 6))
plt.plot(yearly_stats['Year'], yearly_stats['NOC'], 'o-', color='navy')
plt.fill_between(yearly_stats['Year'], yearly_stats['NOC'], alpha=0.2, color='navy')

plt.title('Number of Participating Countries in Olympic Games (1896-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Countries')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(yearly_stats['Year'], rotation=45)
plt.tight_layout()
plt.savefig('participating_countries.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Calculate growth rates
yearly_stats['Total_pct_change'] = yearly_stats['Total'].pct_change() * 100
yearly_stats['NOC_pct_change'] = yearly_stats['NOC'].pct_change() * 100

print("\nChanges in medal counts and participating countries:")
print(yearly_stats[['Year', 'Total_pct_change', 'NOC_pct_change']])

# 3. Calculate period statistics
def calculate_period_stats(df, start_year, end_year):
    period_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    return {
        'Period': f'{start_year}-{end_year}',
        'Avg_Countries': period_data['NOC'].mean(),
        'Avg_Total_Medals': period_data['Total'].mean(),
        'Avg_Gold_Ratio': (period_data['Gold'] / period_data['Total']).mean() * 100
    }

periods = [
    (1896, 1912),  # Early period
    (1920, 1952),  # War and post-war period
    (1956, 1988),  # Cold War period
    (1992, 2024)   # Modern period
]

period_stats = pd.DataFrame([calculate_period_stats(yearly_stats, start, end)
                           for start, end in periods])

print("\nAverage statistics by period:")
print(period_stats)

# Figure 3: Period comparison
plt.figure(figsize=(15, 6))
bars = plt.bar(period_stats['Period'], period_stats['Avg_Total_Medals'],
               color=['lightblue', 'lightgreen', 'lightsalmon', 'lightgray'])

plt.title('Average Number of Medals by Historical Period')
plt.xlabel('Period')
plt.ylabel('Average Number of Medals')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('period_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Identify significant changes
significant_changes = yearly_stats[abs(yearly_stats['Total_pct_change']) > 20]
print("\nYears with significant changes in medal count (>20% change):")
print(significant_changes[['Year', 'Total', 'Total_pct_change']])

# 5. Additional analysis: Calculate compound annual growth rate (CAGR)
start_year = yearly_stats.iloc[0]
end_year = yearly_stats.iloc[-1]
years_diff = end_year['Year'] - start_year['Year']

cagr_medals = (pow(end_year['Total'] / start_year['Total'], 1/years_diff) - 1) * 100
cagr_countries = (pow(end_year['NOC'] / start_year['NOC'], 1/years_diff) - 1) * 100

print("\nLong-term growth analysis:")
print(f"Medals CAGR: {cagr_medals:.2f}%")
print(f"Participating countries CAGR: {cagr_countries:.2f}%")