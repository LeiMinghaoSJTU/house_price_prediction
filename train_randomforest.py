"""
RandomForest模型训练
使用随机森林回归器进行房价预测
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='output'):
    """加载预处理后的数据"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    return X_train, y_train


def train_randomforest_model(X_train, y_train):
    """
    训练随机森林回归模型
    
    RandomForest (随机森林) 是一种基于Bagging (Bootstrap Aggregating) 的集成学习算法。
    它通过并行训练多棵决策树，每棵树使用不同的自助采样数据子集和随机特征子集，
    最终将所有树的预测结果取平均得到最终预测。
    """
    print("\n" + "=" * 50)
    print("训练RandomForest模型...")
    print("=" * 50)
    
    # 创建随机森林回归器
    model = RandomForestRegressor(
        n_estimators=300,        # 树的数量
        max_depth=20,            # 树的最大深度
        min_samples_split=5,     # 内部节点再划分所需最小样本数
        min_samples_leaf=2,      # 叶子节点最小样本数
        max_features='sqrt',     # 每棵树随机选择的特征数量（sqrt(n_features)）
        bootstrap=True,          # 是否使用自助采样
        oob_score=True,          # 是否使用袋外样本评估
        random_state=42,         # 随机种子
        n_jobs=-1,               # 使用所有CPU核心
        verbose=0
    )
    
    # 使用交叉验证评估模型
    print("\n进行5折交叉验证...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"交叉验证RMSE (log尺度): {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # 在全量数据上训练最终模型
    print("\n在全量数据上训练最终模型...")
    model.fit(X_train, y_train)
    
    # 计算训练集上的RMSE
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    print(f"训练集RMSE (log尺度): {train_rmse:.4f}")
    
    # 袋外分数
    if hasattr(model, 'oob_score_') and model.oob_score_:
        oob_pred = model.oob_prediction_
        oob_rmse = np.sqrt(np.mean((oob_pred - y_train) ** 2))
        print(f"袋外样本RMSE (log尺度): {oob_rmse:.4f}")
    
    # 特征重要性（前20个）
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    print("\nTop 20 重要特征:")
    for i, idx in enumerate(top_indices[:20], 1):
        print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    return model


def save_model(model, output_dir='model'):
    """保存训练好的模型"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'randomforest_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n模型已保存到: {model_path}")
    return model_path


def main():
    """主函数"""
    print("=" * 50)
    print("RandomForest房价预测模型训练")
    print("=" * 50)
    
    # 加载预处理后的数据
    print("\n加载预处理后的数据...")
    X_train, y_train = load_preprocessed_data()
    print(f"训练集形状: {X_train.shape}")
    print(f"目标变量范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # 训练模型
    model = train_randomforest_model(X_train, y_train)
    
    # 保存模型
    save_model(model)
    
    print("\n" + "=" * 50)
    print("RandomForest模型训练完成!")
    print("=" * 50)
    
    return model


if __name__ == '__main__':
    main()
