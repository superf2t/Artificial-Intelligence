# 利用GridSearchCV遍历多种参数
from sklearn import svm
from sklearn import model_selection
from sklearn.datasets import load_iris

parameters = {"kernel": ("linear", "rbf"), "C": range(1, 100)}
iris = load_iris()
svr = svm.SVR(gamma="scale")
clf = model_selection.GridSearchCV(svr, parameters, cv=5)
# 运行网格搜索
clf.fit(iris.data, iris.target)
print(clf.best_params_, clf.scorer_, clf.best_score_)

