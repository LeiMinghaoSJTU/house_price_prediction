"""
XGBoost模型训练
使用XGBoost回归器进行房价预测（含超参数网格搜索）
"""

import numpy as np
import pickle
import os
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，尝试安装...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb
    XGBOOST_AVAILABLE = True


def load_preprocessed_data(data_dir='output'):
    """加载预处理后的数据"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    return X_train, y_train


def train_xgboost_model(X_train, y_train):
    """
    训练XGBoost回归模型（含超参数网格搜索）

    XGBoost (eXtreme Gradient Boosting) 是一种基于梯度提升决策树的集成学习算法。
    它通过串行训练多棵决策树，每棵树学习前面所有树的残差（梯度），
    最终将所有树的预测结果相加得到最终预测。

    """
    print("\n" + "=" * 50)
    print("训练XGBoost模型（含超参数网格搜索）...")
    print("=" * 50)
    
    # 1. 定义超参数搜索网格（可根据需求调整，网格越大耗时越长）
    # param_grid = {
    #     'n_estimators': [300, 500, 700],          # 树的数量
    #     'max_depth': [20, 40, 60],                # 树的最大深度
    #     'learning_rate': [0.01, 0.05, 0.1],       # 学习率
    #     'subsample': [0.7, 0.8, 0.9],             # 样本采样比例
    #     'gamma': [0.0, 0.1, 0.2],                 # 节点分裂所需的最小损失减少
    #     'min_child_weight': [1, 3, 5],            # 叶子节点最小样本权重和
    #     'reg_alpha': [0.01, 0.1, 1.0],            # L1正则化
    #     'reg_lambda': [0.1, 1.0, 10.0]            # L2正则化
    # }
    param_grid = {
    'learning_rate': [0.08],               # 学习率
    'max_depth': [4],             
    'n_estimators': [350],                 # 树的数量
    'reg_alpha': [0.01],                   # L1正则化
    'reg_lambda': [8],                     # L2正则化
    'subsample': [0.9],          # 样本采样
    'colsample_bytree': [0.6],   # 特征采样
    'gamma': [0],               # 分裂阈值
    'min_child_weight': [2]          # 叶子节点权重
}
    # 2. 创建基础XGBoost回归器
    base_model = xgb.XGBRegressor(
        colsample_bytree=0.8,    # 特征采样比例（固定）
        colsample_bylevel=0.8,   # 每层的特征采样比例（固定）
        random_state=42,         # 随机种子
        n_jobs=-1,               # 使用所有CPU核心
        objective='reg:squarederror'  # 回归目标函数
    )
    
    # 3. 执行网格搜索（5折交叉验证）
    print("\n开始网格搜索超参数调优（可能耗时较长）...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',  # 负MSE（GridSearchCV默认最大化评分）
        n_jobs=-1,                         # 并行运行（使用所有CPU）
        verbose=1                          # 显示搜索进度
    )
    
    # 拟合数据进行参数搜索
    grid_search.fit(X_train, y_train)
    
    # 4. 提取最优结果
    best_params = grid_search.best_params_
    best_cv_mse = -grid_search.best_score_
    best_cv_rmse = np.sqrt(best_cv_mse)
    
    print("\n" + "-" * 50)
    print("网格搜索最优结果:")
    print(f"最优超参数: {best_params}")
    print(f"最优交叉验证RMSE (log尺度): {best_cv_rmse:.4f}")
    print("-" * 50)
    
    # 5. 获取最优模型
    best_model = grid_search.best_estimator_
    
    # 6. 在全量数据上训练最优模型（确保模型收敛）
    print("\n在全量数据上训练最终最优模型...")
    best_model.fit(X_train, y_train)
    
    # 7. 计算训练集上的RMSE
    train_pred = best_model.predict(X_train)
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    print(f"训练集RMSE (log尺度): {train_rmse:.4f}")
    
    # 8. 特征重要性（前20个）
    feature_importance = best_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    print("\nTop 20 重要特征:")
    for i, idx in enumerate(top_indices[:20], 1):
        print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    return best_model, best_params, best_cv_rmse


def save_model(model, output_dir='model', extra_info=None):
    """保存训练好的模型（含最优参数信息）"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'xgboost_best_model.pkl')
    
    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 保存最优参数信息
    if extra_info:
        params_path = os.path.join(output_dir, 'best_params.txt')
        with open(params_path, 'w') as f:
            f.write("最优超参数:\n")
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n最优交叉验证RMSE (log尺度): {extra_info['best_cv_rmse']:.4f}")
    
    print(f"\n最优模型已保存到: {model_path}")
    if extra_info:
        print(f"最优参数信息已保存到: {params_path}")
    return model_path


def main():
    """主函数"""
    print("=" * 50)
    print("XGBoost房价预测模型训练（含超参数调优）")
    print("=" * 50)
    
    # 加载预处理后的数据
    print("\n加载预处理后的数据...")
    X_train, y_train = load_preprocessed_data()
    print(f"训练集形状: {X_train.shape}")
    print(f"目标变量范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # 训练模型（含超参数搜索）
    model, best_params, best_cv_rmse = train_xgboost_model(X_train, y_train)
    
    # 整理额外信息
    extra_info = best_params.copy()
    extra_info['best_cv_rmse'] = best_cv_rmse
    
    # 保存最优模型
    save_model(model, extra_info=extra_info)
    
    print("\n" + "=" * 50)
    print("XGBoost模型训练（含超参数调优）完成!")
    print("=" * 50)
    
    return model, best_params


if __name__ == '__main__':
    main()