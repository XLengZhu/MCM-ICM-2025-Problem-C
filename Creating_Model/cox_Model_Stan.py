import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt


def train_and_evaluate_cox_model(train_data, test_data):
    """
    训练Cox模型并进行评估

    Parameters:
    -----------
    train_data : DataFrame
        训练数据
    test_data : DataFrame
        测试数据（2020-2024获得首枚奖牌的国家）
    """
    # 1. 训练模型
    print("\n=== Training Cox Model ===")
    cph = CoxPHFitter()
    cph.fit(train_data,
            duration_col='duration',
            event_col='event',
            show_progress=True)

    # 2. 打印模型摘要
    print("\n=== Model Summary ===")
    print(cph.print_summary())

    # 3. 计算训练集和测试集的C-index
    train_c_index = concordance_index(
        train_data['duration'],
        -cph.predict_partial_hazard(train_data),
        train_data['event']
    )

    test_c_index = concordance_index(
        test_data['duration'],
        -cph.predict_partial_hazard(test_data),
        test_data['event']
    )

    print(f"\nC-index (Train): {train_c_index:.3f}")
    print(f"C-index (Test): {test_c_index:.3f}")

    # 4. 可视化实际vs预测生存曲线
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    # 实际生存曲线
    kmf.fit(test_data['duration'],
            event_observed=test_data['event'],
            label='Actual Survival Curve')
    kmf.plot()

    # 预测生存曲线
    cph.predict_survival_function(test_data).mean(axis=1).plot(
        label='Predicted Survival Curve')

    plt.title('Actual vs Predicted Survival Curves')
    plt.xlabel('Time (Olympics)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./curve_stan.png')
    plt.close()

    # 5. 可视化特征重要性
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.savefig('./feature_Importance_stan.png')
    plt.close()

    # 6. 检查比例风险假设
    print("\n=== Testing Proportional Hazards Assumption ===")
    cph.check_assumptions(train_data, show_plots=True)

    return cph, train_c_index, test_c_index

def visualize_country_survival_curves(test_data, cph, duration_col='duration', event_col='event', noc_col='NOC'):
    """
    绘制按国家分组的真实值和预测值的生存曲线。

    Parameters:
    -----------
    test_data : DataFrame
        测试数据集
    cph : CoxPHFitter
        训练好的Cox模型
    duration_col : str
        持续时间的列名
    event_col : str
        事件状态的列名
    noc_col : str
        国家标识的列名（如NOC代码）
    """
    plt.figure(figsize=(14, 10))

    kmf = KaplanMeierFitter()
    countries = test_data[noc_col].unique()

    # 定义颜色映射
    colors = plt.cm.tab20.colors
    color_map = {noc: colors[i % len(colors)] for i, noc in enumerate(countries)}

    for noc in countries:
        # 提取单个国家的数据
        country_data = test_data[test_data[noc_col] == noc]
        if len(country_data) < 5:  # 跳过数据量不足的国家
            continue

        # 绘制真实值生存曲线
        kmf.fit(country_data[duration_col], event_observed=country_data[event_col], label=f"{noc} - Actual")
        kmf.plot_survival_function(ci_show=False, color=color_map[noc], linestyle='--', alpha=0.8)

        # 绘制预测值生存曲线
        predicted_survival = cph.predict_survival_function(country_data)
        predicted_mean_survival = predicted_survival.mean(axis=1)  # 计算平均预测生存概率
        plt.plot(predicted_mean_survival.index, predicted_mean_survival.values,
                 label=f"{noc} - Predicted", color=color_map[noc], linewidth=1.5)

    plt.title('Survival Curves by Country (Actual vs Predicted)', fontsize=16)
    plt.xlabel('Duration (Olympics)', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('./survival_curves_by_country.png', dpi=300)
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 定义要使用的特征列表
    FEATURES = [
        'total_appearances',  # 参赛总次数
        'total_athletes',  # 运动员总数
        'unique_events',  # 参与的项目数
        'unique_sports',  # 参与的大项数
        'athletes_per_games',  # 每届平均运动员数
        'events_per_games',  # 每届平均参与项目数
        'athlete_growth',  # 运动员增长率
        'athlete_growth_last',  # 最近一届的增长率
        'events_per_athlete',  # 每个运动员平均参与项目数
        'duration',
        'event'
    ]
    # 假设我们已经有了划分好的训练集和测试集
    train_data = pd.read_csv("./train_data.csv")
    test_data = pd.read_csv("./test_data.csv")
    train_data = train_data[FEATURES]
    test_data = test_data[FEATURES]
    # 训练和评估模型
    model, train_c_index, test_c_index = train_and_evaluate_cox_model(train_data, test_data)
    print(model,train_c_index,test_c_index)
    visualize_country_survival_curves(test_data, model, duration_col='duration', event_col='event', noc_col='NOC')