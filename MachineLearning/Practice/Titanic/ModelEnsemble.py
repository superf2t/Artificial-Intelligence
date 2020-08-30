from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

train = pd.read_csv("train0.csv")
test = pd.read_csv("test0.csv")
train_df = train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test_feature = test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test_feature)
result = pd.DataFrame({'PassengerId': test['PassengerId'],
                       'Survived': predictions.astype(np.int32)})
result.to_csv("test_ensemble.csv", index=False)