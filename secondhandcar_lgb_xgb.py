import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.callback import EarlyStopping
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb

# 1. 读取数据
train = pd.read_csv('train_data_v1.csv', sep=' ')
test = pd.read_csv('test_data_v1.csv', sep=' ')

# 2. 提取特征和标签
y = train['price']
print("y:", y)
from ipdb import set_trace
set_trace()
X = train.drop(['price', 'SaleID'], axis=1)
X_test = test.drop(['price', 'SaleID'], axis=1)

# 2.1 删除非数值字段
drop_cols = ['regDates', 'creatDates']
for col in drop_cols:
    if col in X.columns:
        X.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)


# 3. 训练验证划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. LightGBM 模型
lgb_model = lgb.LGBMRegressor(n_estimators=10000, learning_rate=0.005)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(50)]
)
lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

# 5. XGBoost 模型
xgb_model = xgb.XGBRegressor(n_estimators=10000, learning_rate=0.005)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# 6. 模型融合（简单平均）
blend_val_pred = 0.5 * lgb_val_pred + 0.5 * xgb_val_pred
blend_test_pred = 0.5 * lgb_test_pred + 0.5 * xgb_test_pred

# 7. 模型评估（使用还原 log1p 的 RMSE）
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("LGBM RMSE:", rmse(y_val, lgb_val_pred))
print("XGBoost RMSE:", rmse(y_val, xgb_val_pred))
print("融合模型 RMSE:", rmse(y_val, blend_val_pred))

# 8. 输出预测结果（还原 log1p）
submission = pd.DataFrame({
    'SaleID': test['SaleID'],
    'price': blend_test_pred
})
submission.to_csv('submission_blend_preprocessed.csv', index=False)
print("✅ 融合预测结果已保存到 submission_blend_preprocessed.csv")
