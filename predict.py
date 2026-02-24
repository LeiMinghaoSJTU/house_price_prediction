"""
测试集预测模块
使用训练好的三个模型对测试集进行预测，并将结果反向转换（expm1）
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def load_data_and_models(data_dir='output', model_dir='model'):
    """加载测试数据和训练好的模型"""
    # 加载测试数据
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
    test_ids = np.load(os.path.join(data_dir, 'test_ids.npy'), allow_pickle=True)
    
    # 加载三个模型
    models = {}
    model_names = ['xgboost_best', 'randomforest', 'lasso']
    
    for name in model_names:
        model_path = os.path.join(model_dir, f'{name}_model.pkl')
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"已加载模型: {name}")
    
    return X_test, test_ids, models


def make_predictions(X_test, models):
    """
    使用三个模型进行预测
    
    注意: 由于训练时目标变量经过了log1p变换，
    预测结果需要使用expm1进行反向变换
    """
    print("\n" + "=" * 50)
    print("开始预测...")
    print("=" * 50)
    
    predictions = {}
    
    for name, model in models.items():
        print(f"\n使用 {name} 模型进行预测...")
        
        # 预测（log尺度）
        log_pred = model.predict(X_test)
        
        # 反向变换：expm1 (即 exp(x) - 1)
        # 因为训练时使用了 log1p (即 log(1 + x))
        pred = np.expm1(log_pred)
        
        predictions[name] = pred
        
        print(f"  预测结果范围: [{pred.min():.2f}, {pred.max():.2f}]")
        print(f"  预测结果均值: {pred.mean():.2f}")
    
    return predictions


def save_predictions(predictions, test_ids, output_dir='output'):
    """保存预测结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    for model_name, pred in predictions.items():
        # 创建DataFrame
        df = pd.DataFrame({
            'Id': test_ids.astype(int),
            'SalePrice': pred
        })
        
        # 按Id排序
        df = df.sort_values('Id').reset_index(drop=True)
        
        # 保存为CSV
        output_path = os.path.join(output_dir, f'{model_name}_submission.csv')
        df.to_csv(output_path, index=False)
        
        saved_files.append(output_path)
        print(f"\n{model_name} 预测结果已保存到: {output_path}")
        print(f"  样本数: {len(df)}")
        print(f"  前5行预览:")
        print(df.head().to_string(index=False))
    
    return saved_files


def generate_ensemble_prediction(predictions, test_ids, output_dir='output'):
    """
    生成集成预测结果（三个模型的平均值）
    这是一个额外的功能，展示如何结合多个模型
    """
    print("\n" + "=" * 50)
    print("生成集成预测（三个模型平均）...")
    print("=" * 50)
    
    # 计算三个模型预测的平均值
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    df = pd.DataFrame({
        'Id': test_ids.astype(int),
        'SalePrice': ensemble_pred
    })
    
    df = df.sort_values('Id').reset_index(drop=True)
    
    output_path = os.path.join(output_dir, 'ensemble_submission.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n集成预测结果已保存到: {output_path}")
    print(f"  预测结果范围: [{ensemble_pred.min():.2f}, {ensemble_pred.max():.2f}]")
    print(f"  预测结果均值: {ensemble_pred.mean():.2f}")
    
    return output_path


def compare_with_sample(output_dir='output'):
    """与样本提交文件进行比较（如果有的话）"""
    sample_path = os.path.join(output_dir, '..', 'upload', 'sample_submission.csv')
    
    if not os.path.exists(sample_path):
        print("\n样本提交文件不存在，跳过比较")
        return
    
    sample_df = pd.read_csv(sample_path)
    
    print("\n" + "=" * 50)
    print("与样本提交文件比较:")
    print("=" * 50)
    print(f"样本提交文件: {sample_path}")
    print(f"  样本数: {len(sample_df)}")
    print(f"  价格范围: [{sample_df['SalePrice'].min():.2f}, {sample_df['SalePrice'].max():.2f}]")
    print(f"  价格均值: {sample_df['SalePrice'].mean():.2f}")


def main():
    """主函数"""
    print("=" * 50)
    print("房价预测 - 测试集预测")
    print("=" * 50)
    
    # 加载数据和模型
    print("\n加载测试数据和模型...")
    X_test, test_ids, models = load_data_and_models()
    print(f"\n测试集形状: {X_test.shape}")
    print(f"测试样本数: {len(test_ids)}")
    
    # 进行预测
    predictions = make_predictions(X_test, models)
    
    # 保存预测结果
    output_dir = 'output'
    save_predictions(predictions, test_ids, output_dir)
    
    # 生成集成预测（可选）
    generate_ensemble_prediction(predictions, test_ids, output_dir)
    
    # 与样本提交文件比较
    compare_with_sample(output_dir)
    
    print("\n" + "=" * 50)
    print("预测完成！结果文件保存在:")
    print(f"  {output_dir}/")
    print("=" * 50)


if __name__ == '__main__':
    main()
