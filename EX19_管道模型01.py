# 使用数据集生成工具生成数据集；
from sklearn.datasets import make_blobs
# 数据集拆分工具
from sklearn.model_selection import train_test_split
# 预处理工具
from sklearn.preprocessing import StandardScaler
#  导入mlp;
from sklearn.neural_network import MLPClassifier
# 画图工具
import matplotlib.pyplot as plt
# 导入管道模型
from sklearn.pipeline import Pipeline
# 导入网格搜索工具；
from sklearn.model_selection import GridSearchCV

# 生成样本数200，分类为2，标准差为5的数据集
X, y = make_blobs(n_samples=200, centers=2, cluster_std=5)
# 拆分数据集；
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)

'''
# 数据集预处理
scaler = StandardScaler().fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)
print('训练集数据形态：', X_train_scale.shape, '\n测试集形态', X_test_scale.shape)
# 画图显示；
# 原始数据集；
plt.scatter(X_train[:, 0], X_train[:, 1])
# 预处理之后的数据集；
plt.scatter(X_train_scale[:, 0], X_train_scale[:, 1], marker='^')
plt.title('training set & scaled training set')
plt.show()
'''
# 使用管道模型对数据进行拟合；一次性完成数据集预处理以及模型拟合操作；
pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=1600, random_state=38))])
pipeline.fit(X_train, y_train)
print('使用管道模型的评分：{:.2f}'.format(pipeline.score(X_test, y_test)))
# 对这个管道模型进行网格搜索；
params = {'mlp__hidden_layer_sizes': [(50,), (100,), (100, 100)], 'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]}
grid = GridSearchCV(pipeline, param_grid=params, cv=3)
grid.fit(X_train, y_train)
print('交叉验证最高得分：{:.2f}'.format(grid.best_score_ * 100))
print('模型最优参数：{}'.format(grid.best_params_))
print('测试集得分：{}'.format(grid.score(X_test, y_test)))
print(pipeline.steps)

