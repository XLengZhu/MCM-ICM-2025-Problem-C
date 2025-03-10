import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

def reconstruct_features(df):
    # 保留显著特征，移除不显著特征
    significant_features = [
        'total_appearances',
        'last_HHI',
        'event_growth_rate',
        'total_appearances_squared',
    ]

    # 添加新的组合特征
    df['hhi_growth'] = df['last_HHI'] * df['event_growth_rate']
    df['relative_growth'] = df['event_growth_rate'] / (df['total_appearances'] + 1)

    return df[significant_features + ['duration', 'event']]

def train_and_evaluate_cox_model_with_probabilities(train_data, test_data):
    """
    训练Cox模型并基于概率评估模型

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
    cph.fit(train_data, duration_col='duration', event_col='event', show_progress=True)

    # 2. 打印模型摘要
    print("\n=== Model Summary ===")
    print(cph.print_summary())

    # 3. 计算生存概率
    def calculate_survival_probabilities(data, model):
        """计算每个数据点的生存概率"""
        survival_probabilities = model.predict_survival_function(data)
        # 提取指定时间点（duration 列）的生存概率
        durations = data['duration'].values
        probabilities = [survival_probabilities.loc[t].iloc[i] for i, t in enumerate(durations)]
        return np.array(probabilities)

    train_probabilities = calculate_survival_probabilities(train_data, cph)
    test_probabilities = calculate_survival_probabilities(test_data, cph)

    # 4. 基于概率评估模型
    train_auc = roc_auc_score(train_data['event'], 1 - train_probabilities)
    test_auc = roc_auc_score(test_data['event'], 1 - test_probabilities)

    train_brier = brier_score_loss(train_data['event'], 1 - train_probabilities)
    test_brier = brier_score_loss(test_data['event'], 1 - test_probabilities)

    print(f"\n=== Model Evaluation ===")
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    print(f"Train Brier Score: {train_brier:.3f}")
    print(f"Test Brier Score: {test_brier:.3f}")

    # 5. 可视化特征重要性
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.savefig('./feature_importance.png')
    plt.close()

    # 6. 检查比例风险假设
    print("\n=== Testing Proportional Hazards Assumption ===")
    cph.check_assumptions(train_data, show_plots=True)

    return cph, train_auc, test_auc, train_brier, test_brier

# 使用示例
if __name__ == "__main__":
    # 定义要使用的特征列表
    FEATURES = [
        'duration',
        'event',
        'total_appearances',
        'total_athletes',
        'unique_events',
        'unique_sports',
        'athletes_per_games',
        'events_per_games',
        'last_HHI',
        'last_athletes_count',
        'last_unique_events',
        'last_unique_sports',
        'last_female_ratio',
        'last_veteran_ratio',
        'athlete_growth_rate',
        'event_growth_rate',
        'historical_events_per_athlete',
        'athlete_per_event'
    ]

    # 加载训练集和测试集
    train_data = pd.read_csv("./cox_train_data.csv")
    test_data = pd.read_csv("./cox_test_data.csv")
    train_data = train_data[FEATURES]
    test_data = test_data[FEATURES]
    # 添加非线性特征
    train_data['total_appearances_squared'] = train_data['total_appearances'] ** 2
    test_data['total_appearances_squared'] = test_data['total_appearances'] ** 2
    # 添加交互特征
    train_data['athlete_event_interaction'] = train_data['last_athletes_count'] * train_data['last_unique_events']
    test_data['athlete_event_interaction'] = test_data['last_athletes_count'] * test_data['last_unique_events']
    train_data = reconstruct_features(train_data)
    test_data = reconstruct_features(test_data)
    # 训练和评估模型
    model, train_auc, test_auc, train_brier, test_brier = train_and_evaluate_cox_model_with_probabilities(train_data, test_data)

    print("\nFinal Model Results:")
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    print(f"Train Brier Score: {train_brier:.3f}")
    print(f"Test Brier Score: {test_brier:.3f}")