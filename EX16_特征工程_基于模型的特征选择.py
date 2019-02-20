import numpy as np
# 导入基于模型的特征选择；
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 导入随机森林分类器；
from sklearn.ensemble import RandomForestRegressor
# 特征缺失修补的工具；
from sklearn.preprocessing import Imputer


stock = pd.read_csv('./Data/20190116.csv', encoding='GBK')
wanted = stock['名称']
# print(stock.head())
# 设置回归分析目标列为“涨幅”
y = stock['涨幅%%']
# print(wanted[y>=10])
# print(y)
#print(y.shape)
# 提取特征从‘现价’到‘量比’之间的全部提取为特征点；
features = stock.loc[:, '现价':'量比']
X = features.values
# 打印一个样本的全部特征点；
# 设置随机森林模型n_estimators参数；
sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=38), threshold='median')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=62)
# 特征缺失值修补；
df_X= Imputer().fit_transform(X_train)
# 数据预处理；
scaler = StandardScaler()
scaler.fit(df_X)
X_train_scaled = scaler.transform(df_X)
X_test_scaled = scaler.transform(X_test)
# 模型特征，随机森林方式拟合；
sfm.fit(X_train_scaled, y_train)
X_train_sfm = sfm.transform(X_train_scaled)
print('基于模型的随机森林拟合后的数据形态：{}'.format(X_train_sfm.shape))
X_test_smf = sfm.transform(X_test_scaled)
mlpr_sfm = MLPRegressor(random_state=62,hidden_layer_sizes=(100,100), alpha=0.001)
mlpr_sfm.fit(X_train_sfm, y_train)
print('采用随机森林模型特征筛选后的拟合结果：{:.2f}'.format(mlpr_sfm.score(X_test_smf, y_test)))

# 导入需要预测的数据集；
Test_stock = pd.read_csv('./Data/testA.csv', encoding='GBK')
# 打印出预测数据的涨幅值；
Test_y = Test_stock['涨幅%%']
# print(Test_y)
# 提取出需要参与预测的特征类型；
Test_features = Test_stock.loc[:, '现价':'量比']
# 用之前拟合好的特征筛选模型筛选一次特征值；
Test_X = Test_features.values
# print(Test_X)
# 预测特征的数据集预处理；
Test_X_scaled = scaler.transform(Test_X)
# 数据特征值筛选；
Test_X_sfm = sfm.transform(Test_X_scaled)
# 数据集预测；
Predict_y = mlpr_sfm.predict(Test_X_sfm)
Test_wanted = Test_stock['名称']

for i, j, k in zip(Test_wanted, Predict_y, Test_y):
    jv = '{:.2f}'.format(j)
    # print('{name}:真实值：{Zhenshi},预测值：{yuce},'.format(name=i, Zhenshi=k, yuce=jv))
    print(i, ": 真实值：", k, ",预测值:", jv)

# 基于模型的特征选择后保留的特征结果。
'''
mask = sfm.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap=plt.cm.cool)
plt.xlabel('Features Selected')
plt.show()
'''

