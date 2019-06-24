import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target
svc = svm.SVC(kernel="linear")
C_s = np.logspace(-10, 0, 10)
scores = []
scores_std = []
for C in C_s:
    svc.C = C
    score_lyst = cross_val_score(svc, X, y, n_jobs=-1)
    scores.append(np.mean(score_lyst))
    scores_std.append(np.std(score_lyst))

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.semilogx(C_s, scores)
plt.semilogx(C_s, np.array(scores)+np.array(scores_std),"r--")
plt.semilogx(C_s, np.array(scores)-np.array(scores_std),"k--")
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()
