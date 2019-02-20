# 导入数据获取工具；
from sklearn.datasets import fetch_lfw_people
# 倒入神经网络模型；
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 导入pca库；
from sklearn.decomposition import PCA

# 载入人脸数据集；
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
image_shape = faces.images[0].shape

# 照片打印出来
fig, axes = plt.subplots(10, 8, figsize=(12, 9), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image, cmap=plt.cm.gray)
ax.set_title(faces.target_names[target])
plt.show()

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(faces.data/255, faces.target, random_state=62)
# 数据白化处理：
pca = PCA(whiten=True, n_components=0.9, random_state=62).fit(X_train)
X_train_whiten = pca.transform(X_train)
X_test_whiten = pca.transform(X_test)
print(X_train_whiten.shape)

mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
mlp.fit(X_train_whiten, y_train)
print('模型准确率：{:.2f}'.format(mlp.score(X_test_whiten, y_test)))

