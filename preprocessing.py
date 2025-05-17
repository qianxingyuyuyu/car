# %%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif']=['simhei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

# %% [markdown]
# ### 1. 数据读取

# %%
train = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('./data/used_car_testB_20200421.csv', sep=' ')

# %% [markdown]
# 所有特征集均脱敏处理
# + SaleID - 交易ID，唯一编码
# + name - 汽车编码
# + regDate - 汽车注册日期，例如20160101，2016年01月01日
# + model - 车型编码
# + brand - 汽车品牌
# + bodyType - 车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
# + fuelType - 燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
# + gearbox - 变速箱：手动：0，自动：1
# + power - 发动机功率：范围 [ 0, 600 ]
# + kilometer - 汽车已行驶公里，单位万km
# + notRepairedDamage - 汽车有尚未修复的损坏：是：0，否：1
# + regionCode - 看车地区编码
# + seller - 销售方：个体：0，非个体：1
# + offerType - 报价类型：提供：0，请求：1
# + creatDate - 汽车上线时间，即开始售卖时间
# + price - 二手车交易价格（预测目标）
# + v系列特征 - v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' 【匿名特征，包含v0-14在内15个匿名特征】

# %%
# 简略查看训练数据
res_train = pd.concat([train.head(),train.tail()])
res_train

# %%
train.shape

# %%
# 简略查看测试数据
res_test = pd.concat([test.head(),test.tail()])
res_test

# %%
test.shape

# %% [markdown]
# ### 2. 数据分析

# %% [markdown]
# #### 2.1 数值属性分析

# %%
# 熟悉训练数据数值属性的相关统计量
# 数据的数量(count)、均值(mean)、标准差(std)、最小值、位于各百分比的值(25%、50%、75%)、最大值 
train_numeric_features = [
    'power', 'kilometer', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
    'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14'
]
train[train_numeric_features].describe()

# %%
# 通过describe()来熟悉测试数据数值属性的相关统计量
test_numeric_features = [
    'power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
    'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14'
]
test[test_numeric_features].describe()

# %%
# 通过info()来熟悉训练数据类型
train.info()

# %%
# 可视化训练数据中的数值属性
# 画所选数据的直方图和KDE曲线
f_train = pd.melt(train, value_vars=train_numeric_features)
g_train = sns.FacetGrid(f_train, col="variable",  col_wrap=3, sharex=False, sharey=False)
g_train = g_train.map(sns.histplot, "value",kde=True)
plt.show()

# %% [markdown]
# 'price'为长尾分布，需要对其做数据转换
# 
# 'power'应该存在异常值，需要处理

# %%
# 可视化测试数据中的数值属性
f_test = pd.melt(test, value_vars=test_numeric_features)
g_test = sns.FacetGrid(f_test, col="variable",  col_wrap=3, sharex=False, sharey=False)
g_test = g_test.map(sns.histplot, "value", kde=True)
plt.show()

# %% [markdown]
# #### 2.2 类别属性分析

# %%
# 分析训练数据类别特征nunique分布
category_features = [
    'name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
    'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'regionCode'
]
for feature in category_features:
    # print(feature + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(feature, train[feature].nunique()))
    # print(train[feature].value_counts())
    print(24 * '-·')

# %%
# 对训练数据类别特征取值较少的，画出直方图
plt.figure(figsize=(15, 12))
i = 1
for feature in category_features:
    if train[feature].nunique() < 50:
        plt.subplot(4, 2, i)
        i += 1
        v = train[feature].value_counts()
        fig = sns.barplot(x=v.index, y=v.values)
        plt.title(feature)
plt.tight_layout()
plt.show()

# %%
# 分析测试数据类别特征nunique分布
for feature in category_features:
    print("{}特征有个{}不同的值".format(feature, test[feature].nunique()))
    print(24 * '-·')

# %%
# 对测试数据类别特征取值较少的，画出直方图
plt.figure(figsize=(15, 12))
i = 1
for feature in category_features:
    if test[feature].nunique() < 50:
        plt.subplot(4, 2, i)
        i += 1
        v = test[feature].value_counts()
        fig = sns.barplot(x=v.index, y=v.values)
        plt.title(feature)
plt.tight_layout()
plt.show()

# %% [markdown]
# 可以发现两个类别特征严重倾斜，分别是'seller'和'offerType'，上述两个特征对分析预测没有任何帮助。
# 
# 另外，'notRepairedDamage'有3个不同的属性值：0.0，1.0，'-'，其中'-'应该也为空值。
# 
# 'name'为汽车交易名称，一般对分析预测没有帮助，但是同名数据不少，可以挖掘

# %% [markdown]
# #### 2.3 数据缺失值

# %%
# 'notRepairedDamage'属性中'-'应该也是空值，用nan替换
train['notRepairedDamage'].replace('-', np.nan, inplace=True)
test['notRepairedDamage'].replace('-', np.nan, inplace=True)

# %%
# 分析训练数据缺失值
train.isnull().sum()[train.isnull().sum() > 0]

# %%
# 训练数据nan可视化
train_missing = train.isnull().sum()
train_missing = train_missing[train_missing > 0]
train_missing.sort_values(inplace=True)
train_missing.plot.bar()
plt.title("训练数据缺省值可视化")
plt.show()

# %%
# 分析测试数据缺失值
test.isnull().sum()[test.isnull().sum() > 0]

# %%
# 测试数据nan可视化
test_missing = test.isnull().sum()
test_missing = test_missing[test_missing > 0]
test_missing.sort_values(inplace=True)
test_missing.plot.bar()
plt.title("测试数据缺省值可视化")
plt.show()

# %% [markdown]
# #### 2.4 相关性分析

# %%
# 对'price'属性进行相关性分析，计算相关系数矩阵
price_numeric = train[train_numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending = False),'\n')

# %%
f, ax = plt.subplots(figsize=(8, 8))
plt.title('Correlation of Numeric Features with Price', y=1, size=16)
sns.heatmap(correlation, square=True, vmax=0.8, linewidths=0.1, cmap=sns.cm.rocket_r)
plt.show()

# %% [markdown]
# 匿名特征v_0, v_3, v_8, v_12与'price'相关性很高

# %% [markdown]
# ### 3. 数据预处理

# %% [markdown]
# #### 3.1 处理目标值长尾分布

# %%
# price 为长尾分布，对该特征进行处理
train['price'] = np.log1p(train['price'])

# 可视化处理后'price'分布
plt.figure(figsize=(10, 8))
sns.histplot(train['price'],kde=True)
plt.show()

# %% [markdown]
# #### 3.2 处理无用值

# %%
# 合并训练数据和测试数据，方便后续数据预处理
df = pd.concat([train, test], axis=0, ignore_index=True)
df.head()

# %%
# SaleID为交易ID，肯定没用，但是我们可以用来统计别的特征的group数量
# name为汽车交易名称，已经脱敏一般没什么好挖掘的，不过同名的好像不少，可以挖掘一下
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
df.drop(['name'], axis=1, inplace=True)

# %% [markdown]
# #### 3.3 处理特征严重倾斜数据

# %%
# 'seller'、'offerType'特征严重倾斜，训练数据中'seller'有一个特殊值，删除该样本
df.drop(df[df['seller'] == 1].index, inplace=True)

df.drop(['seller'], inplace=True, axis=1)
df.drop(['offerType'], inplace=True, axis=1)

# %% [markdown]
# #### 3.4 处理异常值

# %%
# 在题目中规定了power范围为[0, 600]
df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)

# 'notRepairedDamage'属性中'-'应该也是空值，用nan替换
df['notRepairedDamage'].replace('-', np.nan, inplace=True)

# %% [markdown]
# #### 3.5 处理缺失值

# %%
# 查看缺失值
df.isnull().sum()[df.isnull().sum() > 0]

# %%
# 用众数填充缺失值
df.fuelType.fillna(df.fuelType.mode()[0], inplace=True)
df.gearbox.fillna(df.gearbox.mode()[0], inplace=True)
df.bodyType.fillna(df.bodyType.mode()[0], inplace=True)
df.model.fillna(df.model.mode()[0], inplace=True)
df.notRepairedDamage.fillna(df.notRepairedDamage.mode()[0], inplace=True)

# %%
df.isnull().sum()[df.isnull().sum() > 0]

# %% [markdown]
# df是由训练数据和测试数据合并而来，测试数据有50000个样本，预测特征是price，因此df中存在50000个缺失price特征的样本

# %%
# 可视化处理后'power'分布
plt.figure(figsize=(10, 8))
sns.histplot(df['power'],kde=True)
plt.show()

# %% [markdown]
# #### 3.6 处理时间属性信息

# %%
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date

# %%
df['regDates'] = df['regDate'].apply(date_process)
df['creatDates'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDates'].dt.year
df['regDate_month'] = df['regDates'].dt.month
df['regDate_day'] = df['regDates'].dt.day
df['creatDate_year'] = df['creatDates'].dt.year
df['creatDate_month'] = df['creatDates'].dt.month
df['creatDate_day'] = df['creatDates'].dt.day
# df['car_age_day'] = (df['creatDates'] - df['regDates']).dt.days
# df['car_age_year'] = round(df['car_age_day'] / 365, 1)

# %%
# 切割数据，导出数据
output_path = './'
print(df.shape)
train_num = df.shape[0] - 50000
df[:int(train_num)].to_csv(output_path + 'train_data_v1.csv', index=False, sep=' ')
df[train_num:train_num + 50000].to_csv(output_path + 'test_data_v1.csv', index=False, sep=' ')

# %%



