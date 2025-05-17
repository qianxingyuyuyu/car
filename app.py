from flask import Flask, render_template, request, jsonify,flash
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib
import numpy as np
from datetime import datetime
from flask import request, render_template

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

# # 定义数值映射（与训练时保持一致）
# brand_map = {
#     '奥迪': 0,
#     '宝马': 1,
#     '奔驰': 2,
#     '大众': 3,
#     '丰田': 4,
#     '本田': 5,
#     '日产': 6,
#     '别克': 7,
#     '现代': 8,
#     '起亚': 9,
#     '福特': 10,
#     '雪佛兰': 11,
#     '标致': 12,
#     '雪铁龙': 13,
#     '三菱': 14,
#     '马自达': 15,
#     '斯巴鲁': 16,
#     '铃木': 17,
#     '沃尔沃': 18,
#     '英菲尼迪': 19,
#     '捷豹': 20,
#     '路虎': 21,
#     '保时捷': 22,
#     '特斯拉': 23,
#     '荣威': 24,
#     '名爵': 25,
#     '吉利': 26,
#     '长城': 27,
#     '比亚迪': 28,
#     '奇瑞': 29,
#     '江淮': 30,
#     '一汽': 31,
#     '东风': 32,
#     '哈弗': 33,
#     '众泰': 34,
#     '力帆': 35,
#     '红旗': 36,
#     '蔚来': 37,
#     '小鹏': 38,
#     '理想': 39
# }

# bodyType_map = {
#     '豪华轿车': 0,
#     '微型车': 1,
#     '厢型车': 2,
#     '大巴车': 3,
#     '敞篷车': 4,
#     '双门汽车': 5,
#     '商务车': 6,
#     '搅拌车': 7
# }
# fuelType_map = {
#     '汽油': 0,
#     '柴油': 1,
#     '液化石油气': 2,
#     '天然气': 3,
#     '混合动力': 4,
#     '其他': 5,
#     '电动': 6
# }
# gearbox_map = {
#     '手动': 0,
#     '自动': 1
# }
# notRepaired_map = {
#     '是': 1,
#     '否': 0
# }


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}  # 允许的文件类型
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 'your-secret-key' # 必须设置secret_key才能使用flash

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/factors')
def factors():
    return render_template('factors.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'dataFile' not in request.files:
            flash('未选择文件', 'error')
            return render_template('upload.html')

        file = request.files['dataFile']

        if file.filename == '':
            flash('未选择文件', 'error')
            return render_template('upload.html')

        if file and allowed_file(file.filename):
            # 确保上传目录存在
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            # 保存文件
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            flash('文件上传成功！请在"基本信息分析"、"影响因素分析"和"房价预测"模块查看结果。', 'success')
            return render_template('upload.html')

        flash('仅支持CSV格式文件', 'error')

    return render_template('upload.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.get_json()
    selected_type = data.get('type')
    print(selected_type)
    image_map = {
        'gearbox': 'gearbox.png',
        'notRepairedDamage': 'notRepairedDamage.png',
        'kilometer': 'kilometer.png',
        'power': 'power.png',
        'regDate': 'regDate.png',
        'creatDate': 'creatDate.png'
    }
    image_name = image_map.get(selected_type)
    if image_name:
        return jsonify({'image': image_name})
    else:
        return jsonify({'error': 'Invalid selection'}), 400

@app.route('/factors_data', methods=['GET'])
def get_factors():
    # 静态图像名称列表，可根据需要增减
    image_list = [
        'gearbox.png',
        'notRepairedDamage.png',
        'kilometer.png',
        'power.png',
        'regDate.png',
        'creatDate.png',
        'correlation_results.png'
    ]
    return jsonify({'images': image_list})


@app.route('/predict_car', methods=['POST'])
def predict_car():
    import torch
    import joblib
    import numpy as np
    from datetime import datetime

    train_data = pd.read_csv('train_data_v1.csv', sep=' ')
    # 获取表单数据（字符串转 int/float）
    regDate = request.form['regDate']
    brand = int(request.form['brand'])
    bodyType = int(request.form['bodyType'])
    fuelType = int(request.form['fuelType'])
    gearbox = int(request.form['gearbox'])
    power = float(request.form['power'])
    kilometer = float(request.form['kilometer'])
    notRepairedDamage = int(request.form['notRepairedDamage'])

    # 计算 used_time
    reg_datetime = datetime.strptime(regDate, '%Y-%m-%d')
    creat_datetime = datetime(2016, 1, 1)
    used_time = (reg_datetime - creat_datetime).days
    
    # 构造输入数组
    x = np.array([[
        # reg_datetime.year * 365 + reg_datetime.month * 30 + reg_datetime.day,  # 备用
                   brand, bodyType, fuelType, gearbox, power,
                   kilometer, notRepairedDamage, used_time]])
    
    x_30d = np.array([[
        # reg_datetime.year * 365 + reg_datetime.month * 30 + reg_datetime.day,  # 备用
                   brand, bodyType, fuelType, gearbox, power,
                   kilometer, notRepairedDamage, used_time+30]])
    
    
    x_90d = np.array([[
        # reg_datetime.year * 365 + reg_datetime.month * 30 + reg_datetime.day,  # 备用
                   brand, bodyType, fuelType, gearbox, power,
                   kilometer, notRepairedDamage, used_time+90]])

    
    x_180d = np.array([[
        # reg_datetime.year * 365 + reg_datetime.month * 30 + reg_datetime.day,  # 备用
                   brand, bodyType, fuelType, gearbox, power,
                   kilometer, notRepairedDamage, used_time+180]])
    # 标准化
    scaler = joblib.load('scaler.pkl')
    x_scaled = scaler.transform(x)
    x_30d_scaled = scaler.transform(x_30d)
    x_90d_scaled = scaler.transform(x_90d)
    x_180d_scaled = scaler.transform(x_180d)
    # 载入模型
    model = Net(input_dim=x.shape[1])
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()

    train_data['regDate'] = pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce')
    train_data['creatDate'] = pd.to_datetime(train_data['creatDate'], format='%Y%m%d', errors='coerce')

    train_data['used_time'] = (train_data['creatDate'] - train_data['regDate']).dt.days
    train_data.dropna(inplace=True)
    matched_samples = train_data[
    (train_data['brand'] == brand) &
    (train_data['bodyType'] == bodyType) &
    (train_data['fuelType'] == fuelType) &
    (train_data['gearbox'] == gearbox)
    ]
    # 保留4位小数
# 保留4位小数
    matched_samples = matched_samples.copy()
    matched_samples['price'] = matched_samples['price'].apply(lambda x: round(x, 4))
# 调整 regDate 显示年份 +10，格式为 YYYY-MM-DD
    matched_samples['regDate'] = matched_samples['regDate'].apply(
        lambda x: (x + pd.DateOffset(years=10)).strftime('%Y-%m-%d') if pd.notnull(x) else ''
    )

    # 构造展示字段
    matched_preview = matched_samples[['regDate', 'power', 'kilometer', 'used_time', 'price']].head(10).to_dict(orient='records')

    # 添加 used_time 字段到预览数据中
    # matched_preview = matched_samples[['regDate', 'power', 'kilometer', 'used_time', 'price']].head(10).to_dict(orient='records')



    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        x_30d_tensor = torch.tensor(x_30d_scaled, dtype=torch.float32)
        x_90d_tensor = torch.tensor(x_90d_scaled, dtype=torch.float32)
        x_180d_tensor = torch.tensor(x_180d_scaled, dtype=torch.float32)
        # 预测
        pred_log = model(x_tensor).item()
        pred_log_30d = model(x_30d_tensor).item()
        pred_log_90d = model(x_90d_tensor).item()
        pred_log_180d = model(x_180d_tensor).item()
        pred_price = np.expm1(pred_log)
        pred_price_30d = np.expm1(pred_log_30d)
        pred_price_90d = np.expm1(pred_log_90d)
        pred_price_180d = np.expm1(pred_log_180d)
        # 计算预测价格
        
    return render_template('predict.html', prices={
        'today': round(pred_price, 2),
        '30d': round(pred_price_30d, 2),
        '90d': round(pred_price_90d, 2),
        '180d': round(pred_price_180d, 2), 
    },matched_preview=matched_preview)


if __name__ == '__main__':
    app.run(debug=True)