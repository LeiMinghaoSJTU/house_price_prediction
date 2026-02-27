## Kaggle 房价预测项目

本项目是一个基于Kaggle竞赛“House Prices: Advanced Regression Techniques”的房价预测项目。我们使用了三个不同的机器学习模型（XGBoost、RandomForest和Lasso回归）来预测房价。

| 模型             | 原理                       | 交叉验证RMSE (log尺度) |
| ---------------- | -------------------------- | ---------------------- |
| **XGBoost**      | 梯度提升决策树（Boosting） | 0.1272                 |
| **RandomForest** | 随机森林（Bagging）        | 0.1509                 |
| **Lasso**        | 线性回归 + L1正则化        | 0.1254                 |

数据预处理流程见 `preprocessing.ipynb`，模型训练流程见 `train_xgboost.py`、`train_random_forest.py` 和 `train_lasso.py`，预测流程见 `predict.py`。