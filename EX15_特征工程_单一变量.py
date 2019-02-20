import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
# 特征缺失修补的工具；
from sklearn.preprocessing import Imputer
# 导入特征筛选工具；
from sklearn.feature_selection import SelectPercentile

stock = pd.read_csv('./Data/20190116.csv', encoding='GBK')
wanted = stock['名称']
# print(stock.head())
# 设置回归分析目标列为“涨幅”
y = stock['涨幅%%']
print(wanted[y>=10])
# print(y)
#print(y.shape)
# 提取特征从‘现价’到‘量比’之间的全部提取为特征点；
features = stock.loc[:, '现价':'量比']
X = features.values
# 打印一个样本的全部特征点；
# 设置神经网路参数以及数据集拆分
mplr = MLPRegressor(random_state=62 ,hidden_layer_sizes=(100, 100), alpha=0.001)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=62)
# 特征缺失值修补；
df_X= Imputer().fit_transform(X_train)
# 数据预处理；
scaler = StandardScaler()
scaler.fit(df_X)
X_train_scaled = scaler.transform(df_X)
X_test_scaled = scaler.transform(X_test)

# 进行特征筛选；
# 保留50%的特征；
select = SelectPercentile(percentile=50)
select.fit(X_train_scaled, y_train)
X_train_selected = select.transform(X_train_scaled)
X_test_selected = select.transform(X_test_scaled)

print('经过缩放后的特征数量：{}'.format(X_train_scaled.shape))
print('经过筛选后的特征数量：{}'.format(X_train_selected.shape))
mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap=plt.cm.cool)
plt.xlabel('Features Selected')
plt.show()
mplr.fit(X_train_selected, y_train)
print('模型准确率：{:.2f}'.format(mplr.score(X_test_selected, y_test)))
# 数据预处理
'''
mplr.fit(X_train_scaled, y_train)
print('模型准确率：{:.2f}'.format(mplr.score(X_test_scaled, y_test)))

'''

