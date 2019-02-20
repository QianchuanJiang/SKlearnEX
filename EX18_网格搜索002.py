# 导入红酒数据集；
from sklearn.datasets import load_wine
# 套索回归工具；
from sklearn.linear_model import Lasso
# 数据特征集拆分工具
from sklearn.model_selection import train_test_split
import numpy as np
# 导入交叉验证工具
from sklearn.model_selection import cross_val_score
# 导入网格搜索工具
from sklearn.model_selection import GridSearchCV

wine = load_wine()
# 拆分数据集；
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
lasso = Lasso(alpha=1.0, max_iter=100)
# 将需要遍历的参数以及数值写入字典
params = {'alpha': [0.01, 0.1, 1, 10], 'max_iter': [100, 1000, 5000, 10000]}
gird_search = GridSearchCV(lasso, params, cv=6)
gird_search.fit(X_train, y_train)
print('模型最高评分：{:.3f}'.format(gird_search.score(X_test, y_test)))
print('模型最高评分时候的参数值为：{}'.format(gird_search.best_params_))




