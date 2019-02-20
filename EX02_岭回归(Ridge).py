import numpy as np
# 用波士顿房价数据来进行预测；
from sklearn.datasets import load_boston
# 数据集拆分工具；
from sklearn.model_selection import train_test_split
# 导入岭回归数学模型；
from sklearn.linear_model import Ridge

lb=load_boston()
X=lb.data
y=lb.target
# 拆分出训练集和测试集；
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.25)
# 使用岭回归进行拟合；
ridge=Ridge(alpha=5)
ridge.fit(X_train,y_train)
# 输出数据预测得分；
print("================================\n")
print("岭回归训练集得分：{:,.2f}".format(ridge.score(X_train,y_train)))
print("岭回归测试集得分：{:,.2f}".format(ridge.score(X_test,y_test)))
print("================================\n")


