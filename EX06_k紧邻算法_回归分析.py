import numpy as np
#导入回归数据集生成器；
from sklearn.datasets import make_regression
#导入KNN回归分析器；
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
#生成特征数量为1，噪点为50的数据集；
X,y=make_regression(n_features=1,n_informative=1,noise=50,random_state=8)
reg=KNeighborsRegressor()
reg2=KNeighborsRegressor(n_neighbors=2)

#用KNN模型拟合数据
reg.fit(X,y)
reg2.fit(X,y)
z=np.linspace(-3,3,200).reshape(-1,1)
#用散点图将数据可视化；
plt.scatter(X,y,c='orange',edgecolors='k')
plt.plot(z,reg.predict(z),c='blue',linewidth=1)
plt.plot(z,reg2.predict(z),c='red',linewidth=1)
plt.title("KNN Regressor")
#模型评分（保利小数点后面2位）；
print('{:,.2f}'.format(reg.score(X,y)))
print('{:,.2f}'.format(reg2.score(X,y)))
plt.show()

