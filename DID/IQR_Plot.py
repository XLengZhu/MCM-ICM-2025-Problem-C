import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text  # 用于自动调整文本位置避免重叠


def plot_iqr_outliers(sport_data: pd.DataFrame, sport: str, country: str = 'USA'):
    """
    绘制优化后的IQR箱型图和离群点
    """
    # 设置图表样式
    plt.style.use('default')
    sns.set_palette("husl")

    # 计算每年的奖牌数
    yearly_medals = sport_data[sport_data['NOC'] == country].groupby('Year').agg({
        'Medal': lambda x: sum(x != 'No medal')
    }).reset_index()

    # 计算变化量
    changes = []
    for i in range(1, len(yearly_medals)):
        prev_year = yearly_medals.iloc[i - 1]
        curr_year = yearly_medals.iloc[i]

        absolute_change = curr_year['Medal'] - prev_year['Medal']
        relative_change = absolute_change / (prev_year['Medal'] + 1)

        changes.append({
            'Year': curr_year['Year'],
            'Previous_Year': prev_year['Year'],
            'Medals': curr_year['Medal'],
            'Previous_Medals': prev_year['Medal'],
            'Absolute_Change': absolute_change,
            'Relative_Change': relative_change
        })

    changes_df = pd.DataFrame(changes)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'{country} {sport} Performance Changes (1896-2024)',
                 fontsize=16, y=0.95, fontweight='bold')

    # 自定义颜色
    box_color = '#3498db'
    outlier_color = '#e74c3c'
    text_color = '#2c3e50'

    # 绘制两个子图
    for ax, change_type, title in zip(
            [ax1, ax2],
            ['Absolute_Change', 'Relative_Change'],
            ['Absolute Change in Medals', 'Relative Change in Medals']
    ):
        # 绘制箱型图
        sns.boxplot(data=changes_df, y=change_type, ax=ax, color=box_color,
                    width=0.5, showfliers=False)

        # 计算IQR范围
        Q1 = changes_df[change_type].quantile(0.25)
        Q3 = changes_df[change_type].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 找出离群点
        outliers = changes_df[
            (changes_df[change_type] < lower_bound) |
            (changes_df[change_type] > upper_bound)
            ]

        # 绘制离群点
        scatter = ax.scatter(
            x=np.zeros_like(outliers[change_type]),
            y=outliers[change_type],
            color=outlier_color,
            alpha=0.7,
            s=100,
            label='Outliers',
            zorder=5
        )

        # 添加年份标签
        texts = []
        for _, point in outliers.iterrows():
            text = ax.annotate(
                f"{int(point['Year'])}",
                xy=(0, point[change_type]),
                xytext=(20, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                color=text_color,
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
            texts.append(text)

        # 自动调整文本位置避免重叠
        if texts:
            adjust_text(texts, ax=ax)

        # 设置标题和样式
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

        # 设置y轴标签
        ax.set_ylabel(title, fontsize=12)

        # 移除x轴标签
        ax.set_xticks([])

        # 设置背景颜色
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()

    # 保存图片，设置更高的DPI以获得更好的质量
    plt.savefig(f'{sport.lower()}_{country.lower()}_changes.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 打印离群点的详细信息
    print(f"\nOutliers in {sport} for {country}:")
    outliers = changes_df[
        (changes_df['Absolute_Change'] < Q1 - 1.5 * IQR) |
        (changes_df['Absolute_Change'] > Q3 + 1.5 * IQR)
        ].sort_values('Year')

    if len(outliers) > 0:
        print("\nSignificant performance changes:")
        for _, row in outliers.iterrows():
            print(f"\nYear: {int(row['Year'])} (compared to {int(row['Previous_Year'])})")
            print(f"Medals: {int(row['Medals'])} (previous: {int(row['Previous_Medals'])})")
            print(f"Absolute change: {int(row['Absolute_Change'])}")
            print(f"Relative change: {row['Relative_Change']:.2f}")


def main():
    # 读取数据
    athletes_df = pd.read_csv("../summerOly_athletes.csv")

    # 分析排球和体操
    for sport in ['Volleyball', 'Gymnastics']:
        sport_data = athletes_df[athletes_df['Sport'] == sport].copy()
        plot_iqr_outliers(sport_data, sport)


if __name__ == "__main__":
    main()