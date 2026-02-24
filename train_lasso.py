"""
Lasso回归模型训练
使用Lasso回归进行房价预测
"""

import numpy as np
import pickle
import os
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='output'):
    """加载预处理后的数据"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
    return X_train, y_train


def train_lasso_model(X_train, y_train):
    """
    训练Lasso回归模型
    
    Lasso (Least Absolute Shrinkage and Selection Operator) 是一种线性回归方法，
    通过在损失函数中加入L1正则化项来实现特征选择和系数收缩。
    """
    print("\n" + "=" * 50)
    print("训练Lasso回归模型...")
    print("=" * 50)
    
    # 首先使用交叉验证选择最优alpha
    print("\n使用交叉验证选择最优正则化参数alpha...")
    alphas = np.logspace(-4, 1, 50)  # 从0.0001到10的50个值
    
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=5,
        random_state=42,
        n_jobs=-1,
        max_iter=5000,
        tol=1e-4
    )
    
    lasso_cv.fit(X_train, y_train)
    best_alpha = lasso_cv.alpha_
    print(f"最优alpha值: {best_alpha:.6f}")
    
    # 使用最优alpha创建最终模型
    print("\n使用最优alpha训练最终模型...")
    model = Lasso(
        alpha=best_alpha,
        random_state=42,
        max_iter=5000,
        tol=1e-4
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
    model.fit(X_train, y_train)
    
    # 计算训练集上的RMSE
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    print(f"训练集RMSE (log尺度): {train_rmse:.4f}")
    
    # 统计非零系数（被选中的特征）
    n_nonzero = np.sum(model.coef_ != 0)
    n_total = len(model.coef_)
    print(f"\n特征选择结果:")
    print(f"  总特征数: {n_total}")
    print(f"  被选中的特征数: {n_nonzero}")
    print(f"  特征选择比例: {n_nonzero/n_total*100:.2f}%")
    
    # 重要特征（系数绝对值最大的前20个）
    coef_abs = np.abs(model.coef_)
    top_indices = np.argsort(coef_abs)[-20:][::-1]
    print("\nTop 20 重要特征（按系数绝对值）:")
    for i, idx in enumerate(top_indices[:20], 1):
        print(f"  {i}. Feature {idx}: {model.coef_[idx]:.6f}")
    
    return model


def save_model(model, output_dir='model'):
    """保存训练好的模型"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'lasso_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n模型已保存到: {model_path}")
    return model_path


def main():
    """主函数"""
    print("=" * 50)
    print("Lasso回归房价预测模型训练")
    print("=" * 50)
    
    # 加载预处理后的数据
    print("\n加载预处理后的数据...")
    X_train, y_train = load_preprocessed_data()
    print(f"训练集形状: {X_train.shape}")
    print(f"目标变量范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # 训练模型
    model = train_lasso_model(X_train, y_train)
    
    # 保存模型
    save_model(model)
    
    print("\n" + "=" * 50)
    print("Lasso回归模型训练完成!")
    print("=" * 50)
    
    return model


if __name__ == '__main__':
    main()
