import numpy as np
# 导入数据集；
from sklearn.datasets import load_breast_cancer
# 数据集拆分工具
from sklearn.model_selection import train_test_split
# 导入高斯贝叶斯数学模型；
from sklearn.naive_bayes import GaussianNB
# 导入训练曲线方法库；
from sklearn.model_selection import learning_curve
# 导入随机拆分工具；
from sklearn.model_selection import ShuffleSplit
# 画图工具；
import matplotlib.pyplot as plt

cancerData=load_breast_cancer()
X=cancerData.data
y=cancerData.target
# 数据集拆分；
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=38)
# 开始数据拟合；
#gnb=GaussianNB()
#gnb.fit(X_train,y_train)
#print("模型测试评分：{:.4f}".format(gnb.score(X_test,y_test)))
#print("================================\n")
# print(cancerData.keys())
# print(cancerData['DESCR'])
#print("================================\n")


# 对这个数学模型的学习曲线展示；
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    # 设定横坐标标签；
    plt.xlabel("Training examples")
    # 设定纵坐标标签；
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Traning score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt


if __name__=='__main__':
    # 设定图集；
    title = "Learning Curve(Naive Bayes)"
    # 设定拆分数量；
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # 设定模型为高斯朴素贝叶斯；
    estimator = GaussianNB()
    # 调用定义好的画图函数；
    plot_learning_curve(estimator=estimator, title=title, X=X, y=y, ylim=(0.9, 1.01), cv=cv, n_jobs=4)
    plt.show()

