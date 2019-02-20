import pandas as pd
from sklearn.model_selection import train_test_split
# 决策树数学模型；
from sklearn import tree
# 随机森林数学模型；
from  sklearn.ensemble import RandomForestClassifier
import tensorflow as tf


# 用pandas导入数据；
data=pd.read_csv('E:/Works/Ex_SKLearn/Data/adult.csv', header=None, index_col=False, names=['年龄', '单位性质', '权重', '学历', '受教育时长',
                                                                   '婚姻状况', '职业', '家庭情况', '种族', 'Sex',
                                                                   '资产所得', '资产损失', '周工作时长', '原籍', '收入'])
data_lite= data[['年龄', '单位性质', '学历', 'Sex', '周工作时长', '职业', '收入']]
# print(data_lite.head())

# 使用get_dummies将文本数据转化为数值
data_dummies=pd.get_dummies(data_lite)
print(data_dummies.head(5))

#定义数据集的特征；
features= data_dummies.loc[:, '年龄':'职业_ Transport-moving']
X = features.values
# 定义目标数据集，输入大于50K
y = data_dummies['收入_ >50K'].values

X_train,X_text,y_train,y_text=train_test_split(X, y, test_size=0.25)
# go_dating_tree = tree.DecisionTreeClassifier(max_depth=6)
# go_dating_tree.fit(X_train,y_train)
RFclassifier = RandomForestClassifier(n_estimators=10, random_state=8)
RFclassifier.fit(X_train,y_train)
print('测试得分：{:.3f}'.format(RFclassifier.score(X_text, y_text)))



