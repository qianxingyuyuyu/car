import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ✅ 设置设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 1. 读取数据
train = pd.read_csv('train_data_v1.csv', sep=' ')
test = pd.read_csv('test_data_v1.csv', sep=' ')

# ✅ 删除 price ≤ 0 的样本
train = train[train['price'] > 0]
num_brands = train['brand'].nunique()
print(f"共有 {num_brands} 种不同的品牌")
from ipdb import set_trace
set_trace()
# ✅ 添加 used_time 特征（单位：天）
train['regDate'] = pd.to_datetime(train['regDate'], format='%Y%m%d', errors='coerce')
train['creatDate'] = pd.to_datetime(train['creatDate'], format='%Y%m%d', errors='coerce')
test['regDate'] = pd.to_datetime(test['regDate'], format='%Y%m%d', errors='coerce')
test['creatDate'] = pd.to_datetime(test['creatDate'], format='%Y%m%d', errors='coerce')

train['used_time'] = (train['creatDate'] - train['regDate']).dt.days
test['used_time'] = (test['creatDate'] - test['regDate']).dt.days

# ✅ 只选择指定字段
feature_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'used_time']
X = train[feature_cols]
X_test = test[feature_cols]

# ✅ 填充缺失值
X = X.fillna(0)
X_test = X_test.fillna(0)

# ✅ 标签取 log1p
y = np.log1p(train['price'])

# ✅ 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ✅ 划分训练验证集
X_train_scaled, X_val_scaled, y_train_np, y_val_np = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=seed)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# ✅ 构建神经网络模型
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

net = Net(X_train_scaled.shape[1]).to(device)

# ✅ 设置训练参数
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
batch_size = 256
epochs = 100
patience = 10

# ✅ 训练模型
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(epochs):
    net.train()
    for xb, yb in train_loader:
        pred = net(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        val_pred = net(X_val_tensor)
        val_loss = loss_fn(val_pred, y_val_tensor).item()

    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.5f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = net.state_dict()
        import joblib
        torch.save(best_model, 'best_model.pth')
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ 模型参数和标准化器已保存")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("✅ Early stopping triggered.")
            break

# ✅ 加载最优模型并预测
net.load_state_dict(best_model)
net.eval()
with torch.no_grad():
    nn_val_pred = net(X_val_tensor).cpu().numpy().flatten()
    nn_test_pred = net(X_test_tensor).cpu().numpy().flatten()

# ✅ RMSE 评估
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))

print("模型 RMSE:", rmse(y_val_np, nn_val_pred))

# ✅ 输出预测结果
submission = pd.DataFrame({
    'SaleID': test['SaleID'],
    'price': np.expm1(nn_test_pred)
})
submission.to_csv('submission_nn.csv', index=False)
print("✅ 神经网络预测结果已保存至 submission_nn.csv")