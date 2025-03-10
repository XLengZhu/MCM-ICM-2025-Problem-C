import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_iqr_outliers(sport_data: pd.DataFrame, sport: str, country: str = 'USA'):
    """
    绘制IQR箱型图和离群点

    Parameters:
    -----------
    sport_data : DataFrame
        运动项目数据
    sport : str
        运动项目名称
    country : str
        国家代码，默认'USA'
    """
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

    # 创建一个2x1的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{country} {sport} Performance Changes (1896-2024)', fontsize=14)

    # 绘制箱型图和离群点
    for ax, change_type, title in zip(
            [ax1, ax2],
            ['Absolute_Change', 'Relative_Change'],
            ['Absolute Change in Medals', 'Relative Change in Medals']
    ):
        # 绘制箱型图
        sns.boxplot(data=changes_df, y=change_type, ax=ax, color='lightblue')

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
        ax.scatter(
            x=np.zeros_like(outliers[change_type]),
            y=outliers[change_type],
            color='red',
            alpha=0.6,
            s=100,
            label='Outliers'
        )

        # 为离群点添加标签
        for _, point in outliers.iterrows():
            ax.annotate(
                f"{int(point['Year'])}",
                xy=(0, point[change_type]),
                xytext=(0.2, 0),
                textcoords='offset points',
                ha='left',
                va='center'
            )

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{sport.lower()}_{country.lower()}_changes.png')
    plt.close()

    # 打印离群点的详细信息
    print(f"\nOutliers in {sport} for {country}:")
    outliers = changes_df[
        (changes_df['Absolute_Change'] < Q1 - 1.5 * IQR) |
        (changes_df['Absolute_Change'] > Q3 + 1.5 * IQR)
        ]
    print(outliers[['Year', 'Previous_Year', 'Medals', 'Previous_Medals',
                    'Absolute_Change', 'Relative_Change']].to_string())


def main():
    # 读取数据
    athletes_df = pd.read_csv("../summerOly_athletes.csv")

    # 分析排球和体操
    for sport in ['Volleyball', 'Gymnastics']:
        sport_data = athletes_df[athletes_df['Sport'] == sport].copy()
        plot_iqr_outliers(sport_data, sport)


if __name__ == "__main__":
    main()