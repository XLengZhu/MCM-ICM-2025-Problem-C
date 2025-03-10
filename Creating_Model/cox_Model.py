import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

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


def get_predictions(model, prediction_df):
    """
    获取预测结果

    Parameters:
    -----------
    model : CoxPHFitter
        训练好的Cox模型
    prediction_df : DataFrame
        预测数据集

    Returns:
    --------
    results_df : DataFrame
        包含预测概率的结果数据框
    """
    # 计算生存概率
    timeline = np.arange(prediction_df['duration'].max() + 1)  # 确保timeline覆盖所有duration
    survival_prob = model.predict_survival_function(prediction_df, times=timeline)

    # 获取每个国家在其duration时的概率
    probabilities = []
    for idx, row in prediction_df.iterrows():
        duration = row['duration']
        # 找到最接近的时间点
        closest_time = min(timeline, key=lambda x: abs(x - duration))
        prob = 1 - survival_prob.loc[closest_time, idx]
        probabilities.append({
            'Duration': duration,
            'Probability': prob,
            'First_Year': 2028 - duration * 4,
            'Years_Waiting': duration * 4
        })

    # 创建结果数据框
    results_df = pd.DataFrame(probabilities)

    # 按获奖概率排序
    # results_df = results_df.sort_values('Probability', ascending=False)

    return results_df

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

    # 训练和评估模型
    model, train_auc, test_auc, train_brier, test_brier = train_and_evaluate_cox_model_with_probabilities(train_data, test_data)

    print("\nFinal Model Results:")
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    print(f"Train Brier Score: {train_brier:.3f}")
    print(f"Test Brier Score: {test_brier:.3f}")
    predict_data = pd.read_csv("./predict_data.csv")
    # 读取预测数据
    predict_data = pd.read_csv("./predict_data.csv")
    # 保存NOC信息
    noc_info = predict_data['NOC'].copy()
    predict_features = predict_data[FEATURES].copy()
    # 添加特征
    predict_features['total_appearances_squared'] = predict_features['total_appearances'] ** 2
    predict_features['athlete_event_interaction'] = predict_features['last_athletes_count'] * predict_data['last_unique_events']
    # 准备预测数据（只包含模型特征）
    # 进行预测
    result = get_predictions(model, predict_features)
    # 添加NOC信息到结果中
    result['NOC'] = noc_info
    # 重新排序列
    result = result[['NOC', 'Duration', 'Years_Waiting', 'First_Year', 'Probability']]
    result = result.sort_values('Probability', ascending=False)
    # 打印结果
    print("\nPrediction Results:")
    print("\nTop 10 countries most likely to win their first medal:")
    print(result)
    # 保存结果
    result.to_csv("./cox_result.csv", index=False)
