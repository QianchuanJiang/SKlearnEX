from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def mylinear():
    '''
    通过线性回归直接预测波士顿房屋价格；
    :return:None
    '''
    # 获取数据：
    lb = load_boston()
    # 分割数据集和训练集；
    x_train ,x_test , y_train , y_test = train_test_split(lb.data, lb.target,test_size=0.25)

    # 进行标准化处理
    # 特征值和目标值都要进行标准化处理；实例化2个标准化api
    std_x = StandardScaler()
    x_train=std_x.fit_transform(x_train)
    x_test=std_x.transform(x_test)
    # 目标值：
    std_y = StandardScaler()
    y_train=std_y.fit_transform(y_train.reshape(-1,1))
    y_test=std_y.transform(y_test.reshape(-1,1))

    # estimator预测
    # 正规方程求解方式的预测
    #lr = LinearRegression()
    #lr.fit(x_train,y_train)

    #打印权重参数；
    # print(lr.coef_)
    #预测测试集的房价价格,需要把标准化过的数值还原回来。
    #y_predict=std_y.inverse_transform(lr.predict(x_test))

   #print("测试集里面每个房子的预测价格:",y_predict)

    #通过梯度下降的方法进行预测；
    sgd = SGDRegressor()
    sgd.fit(x_train,y_train)

    #打印权重参数；
    print(sgd.coef_)
    #预测测试集的房价价格,需要把标准化过的数值还原回来。
    y_predict=std_y.inverse_transform(sgd.predict(x_test))

    print("测试集里面每个房子的预测价格:",y_predict)
    print("梯度下降法的均方误差：", mean_squared_error(std_y.inverse_transform(y_test),y_predict))

    return None
if __name__=="__main__":
    mylinear()
