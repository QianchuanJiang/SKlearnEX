import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X,iris_y=iris.data,iris.target

print("irisClass: \n{}".format(iris.keys()))
print('============================================')
print(iris["DESCR"])
X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.predict(X_test))
print(y_test)
print('{:.2f}'.format(knn.score(X_test,y_test)))

