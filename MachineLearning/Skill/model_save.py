from sklearn import svm
from sklearn import datasets
import pickle
from sklearn.externals import joblib

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
print(clf)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)


s = pickle.dumps(clf)
clf2 = pickle.loads(s)
y = clf2.predict(X[0:1])
print(y)
# [0]

joblib.dump(clf, 'filename.pkl')
clf = joblib.load('filename.pkl')
print(clf)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

