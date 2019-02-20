import numpy as np
# 导入数据集获取工具；
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
# 导入神经网络分类器
from sklearn.neural_network import MLPClassifier
# 导入图片处理工具；
from PIL import Image

# 加载MNIST手写数字数据集；
mnist = fetch_mldata ('MNIST Original', data_home='./datasets')
print(mnist.data.shape[0])
print(mnist.data.shape[1])
# 把特征值全部处理在0到1之间；
X = mnist.data/255.
y = mnist.target
X_traun, X_test, y_train, y_test= train_test_split(X, y, train_size=5000, test_size=1000, random_state=62)
# 开始拟合模型；
mlp_hw=MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100,100],activation='relu', alpha=1e-5,random_state=62)
mlp_hw.fit(X_traun, y_train)
print('测试集得分：{:.2f}%'.format(mlp_hw.score(X_test,y_test)*100))

image= Image.open('9_ceshi.png').convert('F') 
image= image.resize((28, 28))
arr= []
for i in range(28):
    for j in range(28):
        pixel=1.0- float(image.getpixel((j, i)))/255.
        arr.append(pixel)
arr1 = np.array(arr).reshape(1,-1)
print(mlp_hw.predict(arr1)[0])


