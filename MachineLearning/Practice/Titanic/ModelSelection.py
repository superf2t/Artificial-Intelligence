from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import learning_curve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

train = pd.read_csv("train0.csv")
test = pd.read_csv("test0.csv")
train = train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到LogisticRegression之中
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

print(cross_val_score(clf, X, y, cv=5))

test_feature = test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predict_data = clf.predict(test_feature)
result = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data.astype(np.int32)
})
result.to_csv("ressult.csv", index=False)

train_sizes = np.linspace(.05, 1, 20)
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y,cv=None, n_jobs=1, train_sizes=train_sizes
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.figure()
plt.title(u"学习曲线")
plt.xlabel(u"训练样本数")
plt.ylabel(u"得分")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                 alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                 alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
plt.legend(loc="best")
plt.show()
