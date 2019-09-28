import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from PlotROCCurve import plot_roc_curve
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from MyEnsemble import AverageEnsemble, StackingClassifier, BlendingClassifier, StackingAveragedModels
import xgboost as xgb
from sklearn.preprocessing import Binarizer
from PlotLearningCurve import plot_learning_curve

column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
cancer = pd.read_csv("./breast-cancer-wisconsin.data",names=column_name)
print(cancer.head())
print(cancer.info())

#缺失值处理
cancer = cancer.replace(to_replace="?", value=np.nan)
cancer = cancer.dropna()
print(cancer.shape)
print(cancer.info())
print(cancer.dtypes)
cancer[['Bare Nuclei']] = cancer[['Bare Nuclei']].astype(int)
cancer = cancer.drop('Bare Nuclei', axis=1)
binarizer = Binarizer(threshold=3)
cancer[['Class']] = binarizer.fit_transform(cancer[['Class']])
print(cancer.dtypes)
# pd.get_dummies(cancer[column_name[1:-1]])

#标准化处理
transfer = StandardScaler()
cancer[['Sample code number']] = transfer.fit_transform(cancer[['Sample code number']])

all_data_na = (cancer.isnull().sum() / len(cancer)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
print(missing_data.head(20))

#划分数据集
x = cancer.iloc[:,1:-2]
y = cancer.iloc[:, -1]
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(x_train)



y_predicts = []
model_names = []
# 模型训练
# 创建一个逻辑回归估计器
estimator0 = LogisticRegression()
# 训练模型，进行机器学习
estimator0.fit(x_train, y_train)
# 得到模型，打印模型回归系数，即权重值
print("logist回归系数为:\n",estimator0.coef_)
# 模型评估
# 方法1：真实值与预测值比对
y_predict = estimator0.predict(x_test)
model_names.append('LR')
y_predicts.append(y_predict)

estimator1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4)
estimator1.fit(x_train, y_train)
y_predict = estimator1.predict(x_test)
model_names.append('LGB')
y_predicts.append(y_predict)

estimator2 = GaussianNB()
estimator2.fit(x_train, y_train)
y_predict = estimator2.predict(x_test)
model_names.append('NB')
y_predicts.append(y_predict)

estimator3 = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread=-1)
estimator3.fit(x_train, y_train)
y_predict = estimator3.predict(x_test)
model_names.append('XGB')
y_predicts.append(y_predict)

estimatorx0 = AverageEnsemble(models=[estimator0, estimator1, estimator2])
estimatorx0.fit(x_train, y_train)
y_predict = estimatorx0.predict(x_test)
model_names.append('Ensemble')
y_predicts.append(y_predict)

y_predict = BlendingClassifier(clfs=[estimator0, estimator1, estimator2], train_x=x_train, train_y=y_train, test=x_test)
model_names.append('Blending')
y_predicts.append(y_predict)

# y_predict = StackingClassifier(clfs=[estimator1, estimator2], train_x=x_train, train_y=y_train, test_x=x_test)
# model_names.append('Stacking')
# y_predicts.append(y_predict)

estimatorx1 = StackingAveragedModels(base_models=[estimator0, estimator1, estimator2], meta_model=LogisticRegression())
estimatorx1.fit(x_train, y_train)
y_predict = estimatorx1.predict(x_test)
model_names.append('Stacking')
y_predicts.append(y_predict)


print("预测值为:\n", y_predict)
print("真实值与预测值比对:\n",y_predict == y_test)
# 方法2：计算准确率
print("直接计算准确率为:\n", estimator0.score(x_test, y_test))

y_predict = np.where(y_predict > 0.5, 1, 0)  # 大于2就变为1，否则变为0
#打印精确率、召回率、F1 系数以及该类占样本数
print("精确率与召回率为:\n",classification_report(y_test, y_predict, labels=[0, 1], target_names=["良性","恶性"]))

###模型评估
#ROC曲线与AUC值
roc_auc = roc_auc_score(y_test, y_predict)
print("AUC值:\n",roc_auc)

fp, tp, thresholds = roc_curve(y_test, y_predict)
print(fp)

plt = plot_roc_curve(y_test, y_predicts, model_names)
plt.show()

plt = plot_learning_curve(estimator3, "TT", x_train, y_train)
plt.show()



