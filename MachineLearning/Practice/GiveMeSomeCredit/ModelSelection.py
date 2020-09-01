import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.svm import libsvm
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('cs-training0.csv')
test = pd.read_csv('cs-test0.csv')

X = train.iloc[:, 2:].values
y = train['SeriousDlqin2yrs'].values

#逻辑回归分类器
params ={ "C" : [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
    'penalty' : ['l1', 'l2']
}
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
              random_state=0)
clf = RandomizedSearchCV(estimator=logistic, param_distributions=params, scoring='roc_auc')
search = clf.fit(X, y)
print("Score:", search.best_score_)
print("Score:", search.best_params_)