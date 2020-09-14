import numpy as np
import matplotlib.pyplot as plt
import math


new_ticks1 = np.linspace(-3, 5, 5)
print(new_ticks1.shape)
print(np.percentile(range(1, 101), 75))

import random
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
#用随机数产生一个二维数组。分别是年龄的性别。
df=pd.DataFrame({'Age':np.random.randint(0,70,100),
                'Sex':np.random.choice(['M','F'],100),
                })
#用cut函数对于年龄进行分段分组，用bins来对年龄进行分段，左开右闭
age_groups=pd.cut(df['Age'],bins=[0,18,35,55,70,100])
# print(age_groups)
print(df.groupby(age_groups).count())
print(df.groupby(age_groups).sum())

bins = [0, 20, 40, 60, 100]
labels = [1, 2, 3, 4]
df['AgeRange'] = pd.cut(df['Age'], bins, labels=labels)
print(df.head())

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm

X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)

display = metrics.plot_roc_curve(clf, X_test, y_test)
print('type(display):', type(display))
plt.show()

y_predict = clf.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, plot_roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label="ROC curve(%0.2f)" %roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plt.figure()
plt.title("learning cure")
plt.xlabel('Training examples')
plt.ylabel('Score')
train_sizes, train_scores, test_scores = learning_curve(
    estimator=clf, X=X, y=y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5)
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color='g')
plt.plot(train_sizes, test_scores_mean, 'o-', color='r',
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
         label='Cross-validation score')
plt.legend(loc='best')
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid = {"penalty": ['l1', 'l2'],
              'C': [0.1, 0.5, 1.0]}
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="liblinear")
gs =GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,)
gs.fit(X, y)
print("LR:  ")
print(gs.best_params_)
print(gs.best_score_)

param_grid = {"kernel": ['rbf', 'sigmoid', 'poly'],
              'C': [0.1, 0.5, 1.0]}

clf = svm.SVC(random_state=0)
gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,)
gs.fit(X, y)
print("SVC:  ")
print(gs.best_params_)
print(gs.best_score_)

from xgboost import XGBClassifier
param_grid = {"max_depth": range(3, 10),
              'gamma': [i / 10.0 for i in range(0, 5)],
              'min_child_weight': range(1, 6, 1)}

clf = XGBClassifier(random_state=0)
gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5,)
gs.fit(X, y)
print("XGBoost:  ")
print(gs.best_params_)
print(gs.best_score_)

