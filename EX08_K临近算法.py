import numpy as np
#导入数据生成器；
from sklearn.datasets import make_blobs
#导入KNN分类器；
from sklearn.neighbors import KNeighborsClassifier
#导入画图工具；
import matplotlib.pyplot as plt

#生成样本；
data=make_blobs(n_samples=500, centers=5, random_state=8)
X, y=data
#将数据集可视化；
clf=KNeighborsClassifier()
#X为数据集，一个二维数组，y是目标值；
clf.fit(X, y)
#画图，求出坐标轴的范围；
x_min, x_max=X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max=X[:, 1].min()-1, X[:, 1].max()+1
xx ,yy =np.meshgrid(np.arange(x_min,x_max, .02), np.arange(y_min,y_max,.02))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.spring)
#生成散点图
plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.spring,edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
plt.scatter(6.75,4.82,marker="*",c='blue',s=200)
plt.show()


