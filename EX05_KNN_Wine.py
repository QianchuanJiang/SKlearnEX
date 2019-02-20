import numpy as np
#导入红酒数据集；
from sklearn.datasets import load_wine
#导入数据集分类方法；
from sklearn.model_selection import train_test_split
#导入KNN模型分类器；
from sklearn.neighbors import KNeighborsClassifier
# 初始化KNN分类器；
knn = KNeighborsClassifier(n_neighbors=1)
# 初始化红酒分类数据；
wine_dataset = load_wine()

'''
训练与测试的方法；
'''
def Knn_wines():
    # 打印红酒数据集中的各键；
    '''
    print('代码运行结果：')
    print('=================================')
    print("红酒数据集中的键：\n{}".format(wine_dataset.keys()))
    print('=================================')
    print('/n/n')
    print('数据概况：{}'.format(wine_dataset['data'].shape))
    print('=================================')
    print('/n/n')
    print(wine_dataset['DESCR'])
    '''
    # 将数据集拆分成训练集和测试集；
    X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'], wine_dataset['target'], random_state=0)
    print('代码运行结果：')
    print('=================================')
    # 训练特征；
    print("训练特征：\n{}".format(X_train.shape))
    # 测试特征；
    print("测试特征：\n{}".format(X_test.shape))
    # 训练目标值；
    print("训练目标值：\n{}".format(y_train.shape))
    # 测试目标值；
    print("测试目标值：\n{}".format(y_test.shape))
    print('=================================')
    #用数据模型对数据进行拟合；
    knn.fit(X_train,y_train)
    print(knn)
    print('\n')
    print('=================================')
    #导入测试集得出测试分数；
    print('测试数据集得分:{:.2f}'.format(knn.score(X_test,y_test)))
    print('=================================\n')

'''
预测函数，输入指定的预测值；
'''
def RunClassifierFun(X_data):
    #使用.predict进行预测；
    prediction=knn.predict(X_data)
    print('\n\n')
    print("预测新红酒分类为：{}".format(wine_dataset['target_names'][prediction]))
    print('=================================\n')


if __name__=='__main__':
    Knn_wines()
    NewData=np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820], [3.2, 0.77, 2.51, 14.5, 96.6, 1.04, 2.55, 0.57, 0.47, 6.2, 1.05, 3.33, 220]])
    RunClassifierFun(NewData)
