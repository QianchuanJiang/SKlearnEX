import pandas as pd
#划分训练样本和测试样本的包
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knncls():
    '''
    K-近邻预测用户签到位置
    :return:
    '''
    #读取数据：
    data=pd.read_csv("Data/train.csv")
    # print(data.head(10))


    #处理数据：
    #缩小数据集；
    data=data.query("x>1 & x<1.5&y>2.5&y<3")
    #处理时间的数据；
    time_Value= pd.to_datetime(data['time'],unit='s')

    #构造特征 把日期格式转换成字典格式
    time_Value=pd.DatetimeIndex(time_Value)

    #构造时间特征：
    data['day']=time_Value.day
    data['weekDay']=time_Value.weekday
    data['hour']=time_Value.hour
    #把时间戳特征删除,pandas中，参数1，表示列，0表示行
    data=data.drop(['time'],axis=1)

    #把签到数量少于n个的目标位置删除；
    place_count=data.groupby('place_id').count()
    #'reset_index',重新排序样本
    tf=place_count[place_count.row_id>5].reset_index()
    data=data[data['place_id'].isin(tf.place_id)]
    #删除row_Id列；
    data=data.drop(['row_id'],axis=1)

    #取出数据当中的特征值和目标值；
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    #进行数据的分割训练集合测试集，返回值顺序注意：训练特征值，训练目标值，测试特征值，测试目标值；
    x_train, x_test, y_train, y_text = train_test_split(x, y, test_size=0.25)
    print(x_train)
    #特征工程(标准化)：
    std=StandardScaler()

    #对测试集和训练集的特征值进行标准化；
    x_train=std.fit_transform(x_train)
    x_test=std.transform(x_test)

    #进行算法流程：
    knn = KNeighborsClassifier(n_neighbors=6)
    #传入训练样本
    knn.fit(x_train,y_train)
    #得出预测结果
    y_predict=knn.predict(x_test)
    print("预测的目标签到位置为：",y_predict)

    #得出预测的准确率
    print("预测的准确率：", knn.score(x_test, y_text))
    return None

if __name__ =="__main__":
    knncls()


