import jieba
# 导入向量化工具；
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()


cn = jieba.cut('那只敏捷的棕色狐狸跳过了一只懒惰的狗')
cn = [' '.join(cn)]
vect.fit(cn)
print(vect.vocabulary_)