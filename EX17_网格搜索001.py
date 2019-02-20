# 导入红酒数据集；
from sklearn.datasets import load_wine
# 套索回归工具；
from sklearn.linear_model import Lasso
# 数据特征集拆分工具
from sklearn.model_selection import train_test_split
import numpy as np
# 导入交叉验证工具
from sklearn.model_selection import cross_val_score

wine = load_wine()
# 拆分数据集；
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)

# 设置最佳评分值为0；
best_score = 0
# 设置alpha的遍历值为：0.01,0.1,1,10
for alpha in [0.01,0.1,1,10]:
    # max_iter的数值为100,1000,5000,10000
    for max_iter in [100,1000,5000,10000]:
        lasso = Lasso(alpha=alpha, max_iter=max_iter)
        scores = cross_val_score(lasso, X_train, y_train, cv=6)
        score = np.mean(scores)
        lasso.fit(X_train,y_train)
        score = lasso.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters={'alpha':alpha,'最大迭代次数': max_iter}
            # 打印结果：
            print('模型的最高评分为：{:.3f}'.format(best_score))
            print('模型最高评分时的参数为：{}'.format(best_parameters))


