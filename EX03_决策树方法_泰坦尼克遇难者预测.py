# _*_ coding:utf-8 _*_
import pandas as pd
#划分训练样本和测试样本的包
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pydot
import os

def decision():
    '''
    决策树的方法来对泰坦尼克号乘船人员名单进行生死预测
    :return:
    '''
    # 获取数据
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据。找出特征和目标值；船舱等级，年龄，性别三个维度的数据；
    x = titanic[['pclass', 'age', 'sex']]
    # 载入目标值：
    y = titanic['survived']
    print(y)
    # 处理缺失值,对年龄一栏里面的缺失值取整体年龄的平均值；
    x['age'].fillna(x['age'].mean(), inplace=True)
   # print(x)
    # 分割数据，训练集和测试集
    x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.1)
    # 进行处理（特征工程）特征，类别统一，用 "one_hot"编码的方式；
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))
  #  print(dict.get_feature_names())
   # print(x_train)
    #print(y_train)
    # 用决策树的方法进行预测；
    dec=DecisionTreeClassifier()
    dec.fit(x_train,y_train)
    # max_depth=5, n_estimators=10, max_features=1
    #rf=RandomForestClassifier()
    #rf.fit(x_train,y_train)
    print("预测的准确率：", dec.score(x_test, y_test))
    # 预测的准确率
    #print("预测的准确率：",dec.score(x_test,y_test))
    # 导出决策树的结构；
    os.environ['PATH'] = os.environ['PATH'] + (';c:\\Program Files (x86)\\Graphviz2.38\\bin\\')
    export_graphviz(dec, out_file="./tree.dot" , feature_names=['Age', 'pclass_1st', 'pclass_2st', 'pclass_3st', 'Famale', 'male'])
    (graph,) = pydot.graph_from_dot_file('tree.dot', encoding='UTF-8')
    graph.write_png('tree.png')

    return None

if __name__ == "__main__":
    decision()