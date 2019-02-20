import numpy as np
#归一化的工具集；
from sklearn import preprocessing
#模型测试验证的工具；
from sklearn.cross_validation import train_test_split
#制作数据集的模块；
from sklearn.datasets.samples_generator import make_classification
#支持向量机模型；
from sklearn.svm import SVC
#画图工具；
import matplotlib.pyplot as plt

#生成数据；300个特征。2种属性；
X,y=make_classification(n_samples=300,n_features=2,n_redundant=0,n_informative=2,
                        random_state=22,n_clusters_per_class=1,scale=100)
#数据归一化；
X=preprocessing.scale(X)
#数据集划分；
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf=SVC()
clf.fit(X_train,y_train)
#精确率；
print('{:,.2f}'.format( clf.score(X_test,y_test)))

#画散点图；

plt.scatter(X[:,0], X[:,1],c=y)
plt.show()




