import numpy as np
# 导入数据集生成工具；
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# 导入kmeans工具；
from sklearn.cluster import  KMeans
# 生成分类数为1的数据集；
blobs = make_blobs(n_samples=100, random_state=1, centers=1)
X_blobs = blobs[0]
# 将聚类的数量分成3类；
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_blobs)

# 画图；
x_min, x_max = X_blobs[:, 0].min()-0.5, X_blobs[:, 0].max()+0.5
y_min, y_max = X_blobs[:, 1].min()-0.5, X_blobs[:, 1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, 0.1))
# 将xx, yy这两个数组合成一个二维矩阵；
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.summer, aspect='auto', origin='lower')
# 输出数据点；
plt.plot(X_blobs[:, 0], X_blobs[:, 1], 'r.', markersize=5)
# 用×表示聚类中心；
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=3, color='b', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
print("数据点的分类标签：\n{}".format(kmeans.labels_))
# plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c='r', edgecolors='k')
plt.show()