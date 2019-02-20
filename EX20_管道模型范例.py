# 导入pandas
import pandas as pd
# 预处理工具
from sklearn.preprocessing import StandardScaler
# 导入mlp
from sklearn.neural_network import MLPRegressor
# 导入交叉验证工具
from sklearn.model_selection import cross_val_score
# 导入make_pipline模块
from sklearn.pipeline import make_pipeline
# 导入特征选择模块
from sklearn.feature_selection import SelectFromModel
# 导入随机森林模型
from sklearn.ensemble import RandomForestRegressor

# 导入csv数据；
stocks = pd.read_csv('./Data/20190116.csv', encoding='GBK')
X = stocks.loc[:, '现价':'量比'].values
y = stocks['涨幅%%']
print(X.shape,y.shape)
# 使用交叉验证的方法对模型进行评分；
scores = cross_val_score(MLPRegressor(random_state=38), X, y, cv=3)
print(scores.mean())
# 使用make_pipeline的方法来建立管道模型
pipe = make_pipeline(StandardScaler(), MLPRegressor(random_state=38))
print('\n',pipe.steps)
# 对建立的管道模型进行交叉验证
scores = cross_val_score(pipe, X, y, cv=3)
print('使用数据预处理之后的管道模型得分：{:.2f}'.format(scores.mean()))
# 这个管道流程包括数据的预处理，特征选择，用mlp进行拟合的一套流程；
pipe_SelectModel = make_pipeline(StandardScaler(), SelectFromModel(RandomForestRegressor(random_state=38)), MLPRegressor(random_state=38, max_iter=10000))
pipe_SelectModel.fit(X,y)
print(pipe_SelectModel.steps)
scores = cross_val_score(pipe_SelectModel, X, y, cv=3)
print('使用数据预处理，特征筛选后的管道模型得分：{:.2f}'.format(scores.mean()))


