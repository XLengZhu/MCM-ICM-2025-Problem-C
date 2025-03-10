import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_lgb_model(train_data, test_data, target_col, categorical_features=None):
    """
    训练LightGBM模型
    """
    # 准备特征和目标变量
    feature_cols = [col for col in train_data.columns if col not in
                    ['gold_predict', 'metal_predict', 'NOC_Code', 'Year']]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    # 创建数据集
    train_dataset = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )

    # 设置参数
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # 训练模型
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    return model, metrics, y_pred


def predict_with_intervals(model, X_test, n_iterations=100, confidence_level=0.95):
    """
    使用bootstrap方法生成预测区间
    """
    predictions = []
    for _ in range(n_iterations):
        # 随机选择80%的特征
        feature_mask = np.random.choice([0, 1], size=X_test.shape[1], p=[0.2, 0.8])
        X_bootstrap = X_test.copy()
        X_bootstrap.iloc[:, feature_mask == 0] = 0

        pred = model.predict(X_bootstrap)
        predictions.append(pred)

    # 计算预测区间
    predictions = np.array(predictions)
    lower = np.percentile(predictions, ((1 - confidence_level) / 2) * 100, axis=0)
    upper = np.percentile(predictions, (1 - (1 - confidence_level) / 2) * 100, axis=0)
    mean_pred = np.mean(predictions, axis=0)

    return mean_pred, lower, upper


def plot_feature_importance(model, name='model', top_n=20):
    """
    可视化特征重要性

    Parameters:
    -----------
    model : LightGBM model
        训练好的模型
    name : str
        模型名称（用于标题）
    top_n : int
        显示前n个最重要的特征
    """
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance()
    })

    # 按重要性排序并选择前N个特征
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制水平条形图
    sns.barplot(x='importance', y='feature', data=importance_df,
                palette='viridis')

    # 设置标题和标签
    plt.title(f'Top {top_n} Feature Importance - {name}', fontsize=14, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # 添加网格线
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图形
    plt.savefig(f'{name}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def predict_2028_medals(gold_model,total_model):
    """预测2028年奥运会奖牌情况"""

    # 加载预测数据
    predict_data = pd.read_csv('lgb_predict_dataset.csv')

    # 确保分类特征的一致性
    categorical_features = ['is_host_history', 'ishost']
    for col in categorical_features:
        if col in predict_data.columns:
            predict_data[col] = predict_data[col].astype('category')

    # 准备特征
    feature_cols = [col for col in predict_data.columns if col not in ['NOC_Code', 'Year']]
    X_pred = predict_data[feature_cols]

    # 4. 进行预测并计算预测区间
    def get_predictions_with_intervals(model, X, n_iterations=100, confidence_level=0.95):
        predictions = []
        for _ in range(n_iterations):
            # Bootstrap特征
            feature_mask = np.random.choice([0, 1], size=X.shape[1], p=[0.2, 0.8])
            X_bootstrap = X.copy()
            X_bootstrap.iloc[:, feature_mask == 0] = 0

            pred = model.predict(X_bootstrap)
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, ((1 - confidence_level) / 2) * 100, axis=0)
        upper = np.percentile(predictions, (1 - (1 - confidence_level) / 2) * 100, axis=0)

        return mean_pred, lower, upper

    # 5. 获取预测结果
    gold_mean, gold_lower, gold_upper = get_predictions_with_intervals(gold_model, X_pred)
    total_mean, total_lower, total_upper = get_predictions_with_intervals(total_model, X_pred)

    # 6. 整理结果
    results = pd.DataFrame({
        'NOC': predict_data['NOC_Code'],
        'Predicted_Gold': np.round(gold_mean, 0).astype(int),
        'Gold_Lower_95': np.round(gold_lower, 1),
        'Gold_Upper_95': np.round(gold_upper, 1),
        'Predicted_Total': np.round(total_mean, 1).astype(int),
        'Total_Lower_95': np.round(total_lower, 1),
        'Total_Upper_95': np.round(total_upper, 1)
    })

    # 7. 添加主办国标记
    results['Is_Host'] = (results['NOC'] == 'USA').astype(int)

    # 8. 按预测金牌数排序
    results = results.sort_values('Predicted_Gold', ascending=False)

    # 9. 打印结果摘要
    print("\n2028 Los Angeles Olympics Medal Predictions:")
    print("\nTop 10 countries by predicted gold medals:")
    print(results.head(10).to_string(index=False))

    print("\nHost country (USA) prediction:")
    print(results[results['NOC'] == 'USA'].to_string(index=False))

    print("\nSummary statistics:")
    print(f"Total predicted gold medals: {results['Predicted_Gold'].sum():.0f}")
    print(f"Total predicted medals: {results['Predicted_Total'].sum():.0f}")

    # 10. 保存完整结果
    results.to_csv('predictions_2028_full.csv', index=False)
    print("\nFull predictions saved to 'predictions_2028_full.csv'")

    return results

def main():
    # 读取数据
    train_data = pd.read_csv("./lgb_train_data.csv")
    test_data = pd.read_csv("./lgb_test_data.csv")

    # 定义类别特征（移除非特征列）
    categorical_features = ['is_host_history', 'ishost']

    # 转换类别特征为 category 类型
    for col in categorical_features:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('category')
            test_data[col] = test_data[col].astype('category')

    # 训练金牌预测模型
    gold_model, gold_metrics, gold_pred = train_lgb_model(
        train_data,
        test_data,
        'gold_predict',
        categorical_features
    )
    print("\nGold Medal Prediction Metrics:")
    print(gold_metrics)

    # 训练总奖牌预测模型
    total_model, total_metrics, total_pred = train_lgb_model(
        train_data,
        test_data,
        'metal_predict',
        categorical_features
    )
    print("\nTotal Medal Prediction Metrics:")
    print(total_metrics)

    # 特征重要性可视化
    plot_feature_importance(gold_model, 'Gold_Medal_Model')
    plot_feature_importance(total_model, 'Total_Medal_Model')

    # 保存特征重要性数据
    feature_importance_gold = pd.DataFrame({
        'feature': gold_model.feature_name(),
        'importance': gold_model.feature_importance()
    }).sort_values('importance', ascending=False)
    feature_importance_total = pd.DataFrame({
        'feature': total_model.feature_name(),
        'importance': total_model.feature_importance()
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Important Features for Gold Medal Prediction:")
    print(feature_importance_gold.head(10))
    print("\nTop 10 Important Features for Total Medal Prediction:")
    print(feature_importance_total.head(10))
    # 保存特征重要性到CSV
    feature_importance_gold.to_csv('gold_feature_importance.csv', index=False)
    feature_importance_total.to_csv('total_feature_importance.csv', index=False)

    # 保存预测结果
    predictions = pd.DataFrame({
        'NOC_Code': test_data['NOC_Code'],
        'Year': test_data['Year'],
        'Gold_Actual': test_data['gold_predict'],
        'Gold_Predicted': gold_pred,
        'Total_Actual': test_data['metal_predict'],
        'Total_Predicted': total_pred
    })
    predictions.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")
    result = predict_2028_medals(gold_model, total_model)




if __name__ == "__main__":
    main()