"""
数据预处理模块
对训练集和测试集进行预处理：
1. 分类特征进行one-hot编码
2. 数值特征进行标准化
3. 处理缺失值
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


def load_data(train_path, test_path):
    """加载训练和测试数据"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def identify_feature_types(df):
    """识别分类特征和数值特征"""
    categorical_features = []
    numerical_features = []
    
    for col in df.columns:
        if col in ['id', 'saleprice']:
            continue
        if df[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    return categorical_features, numerical_features


def handle_missing_values(df, categorical_features, numerical_features):
    """处理缺失值"""
    df = df.copy()
    
    # 数值特征：用中位数填充
    for col in numerical_features:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # 分类特征：用众数填充
    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Missing'
            df[col].fillna(mode_val, inplace=True)
    
    return df


def preprocess_data(train_df, test_df, output_dir='.'):
    """
    主预处理函数
    
    Returns:
        X_train: 预处理后的训练特征
        y_train: 训练目标（log1p处理后的）
        X_test: 预处理后的测试特征
        test_ids: 测试集ID
    """
    print("=" * 50)
    print("开始数据预处理...")
    print("=" * 50)
    
    # 分离目标变量
    y_train = train_df['saleprice'].values
    train_df = train_df.drop('saleprice', axis=1)
    
    # 保存测试集ID
    test_ids = test_df['id'].values if 'id' in test_df.columns else None
    if 'id' in test_df.columns:
        test_df = test_df.drop('id', axis=1)
    
    # 识别特征类型
    categorical_features, numerical_features = identify_feature_types(train_df)
    
    print(f"\n分类特征数量: {len(categorical_features)}")
    print(f"数值特征数量: {len(numerical_features)}")
    
    # 处理缺失值
    print("\n处理缺失值...")
    train_df = handle_missing_values(train_df, categorical_features, numerical_features)
    test_df = handle_missing_values(test_df, categorical_features, numerical_features)
    
    # 对分类特征进行one-hot编码
    print("\n对分类特征进行one-hot编码...")
    
    # 合并训练集和测试集以确保编码一致
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # one-hot编码
    combined_encoded = pd.get_dummies(combined_df, columns=categorical_features, drop_first=False)
    
    # 分割回训练集和测试集
    n_train = len(train_df)
    X_train = combined_encoded.iloc[:n_train].copy()
    X_test = combined_encoded.iloc[n_train:].copy()
    
    print(f"one-hot编码后的特征数量: {X_train.shape[1]}")
    
    # 对数值特征进行标准化
    print("\n对数值特征进行标准化...")
    scaler = StandardScaler()
    
    # 只对数值特征进行标准化（排除one-hot编码后的特征）
    numerical_cols_in_encoded = [col for col in X_train.columns 
                                  if any(nf in col for nf in numerical_features) 
                                  and col in numerical_features]
    
    # 更准确地识别数值列：那些在原始数据中是数值类型的列
    numerical_cols_to_scale = []
    for col in X_train.columns:
        # 检查是否是原始数值特征（不是one-hot编码产生的）
        is_original_numerical = col in numerical_features
        if is_original_numerical:
            numerical_cols_to_scale.append(col)
    
    print(f"需要标准化的数值特征数量: {len(numerical_cols_to_scale)}")
    
    if numerical_cols_to_scale:
        # 拟合并转换训练集
        X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        # 转换测试集
        X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])
    
    # 保存预处理器
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = {
        'scaler': scaler,
        'numerical_cols': numerical_cols_to_scale,
        'feature_columns': X_train.columns.tolist()
    }
    
    with open(os.path.join(output_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print("\n预处理完成!")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"目标变量形状: {y_train.shape}")
    
    return X_train, y_train, X_test, test_ids


def main():
    """主函数"""
    # 数据路径
    train_path = 'input/train_processed.csv'
    test_path = 'input/test_processed.csv'
    output_dir = 'output'
    
    # 加载数据
    print("加载数据...")
    train_df, test_df = load_data(train_path, test_path)
    
    print(f"训练集原始形状: {train_df.shape}")
    print(f"测试集原始形状: {test_df.shape}")
    
    # 预处理
    X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df, output_dir)
    
    # 保存预处理后的数据
    print("\n保存预处理后的数据...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train.values)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test.values)
    np.save(os.path.join(output_dir, 'test_ids.npy'), test_ids)
    
    # 保存特征名称
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(X_train.columns.tolist()))
    
    print("\n所有预处理数据已保存!")
    print(f"预处理器保存在: {output_dir}/preprocessor.pkl")
    print(f"特征数据保存在: {output_dir}/X_train.npy, X_test.npy")
    print(f"目标变量保存在: {output_dir}/y_train.npy")
    
    return X_train, y_train, X_test, test_ids


if __name__ == '__main__':
    main()
