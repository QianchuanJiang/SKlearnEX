import numpy as np
# 导入MPL分类器；
from sklearn.neural_network import MLPClassifier
# 导入红酒数据集；
from sklearn.datasets import load_wine
# 导入数据拆分工具；
from  sklearn.model_selection import train_test_split

# 导入画图工具；
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['FF0000', '#00FF00', '0000FF'])


# 数据集初始化；
wine = load_wine()
X = wine.data[:, :2]
y = wine.target
X_train, X_text, y_train, y_test= train_test_split(X, y, random_state=0)
# 初始化分类器；
MLP= MLPClassifier(solver='lbfgs',hidden_layer_sizes=[10, 10],activation='tanh', alpha=1)
MLP.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min()-1, X_train[:, 0].max()+1
y_min, y_max = X_train[:, 1].min()-1, X_train[:, 1].max()+1
# 生成一个二维阵列，间隔为.02；
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
# 把阵列导入到模型中识别出结果；
Z = MLP.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 散点图；

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=60)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("MLPClassifier:solver='lbfgs")
plt.show()

