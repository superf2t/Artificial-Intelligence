import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report

column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
cancer = pd.read_csv("./breast-cancer-wisconsin.data",names=column_name)
print(cancer.head())
print(cancer.info())

#缺失值处理
cancer = cancer.replace(to_replace="?", value=np.nan)
cancer = cancer.dropna()
print(cancer.info())

#数据集划分
x = cancer.iloc[:,1:-2]
y = cancer.iloc[:, -1]
#划分数据集
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3)

#标准化处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 模型训练
# 创建一个逻辑回归估计器
estimator = LogisticRegression()
# 训练模型，进行机器学习
estimator.fit(x_train,y_train)
# 得到模型，打印模型回归系数，即权重值
print("logist回归系数为:\n",estimator.coef_)

# 模型评估
# 方法1：真实值与预测值比对
y_predict = estimator.predict(x_test)
print("预测值为:\n", y_predict)
print("真实值与预测值比对:\n",y_predict == y_test)
# 方法2：计算准确率
print("直接计算准确率为:\n",estimator.score(x_test, y_test))

#打印精确率、召回率、F1 系数以及该类占样本数
print("精确率与召回率为:\n",classification_report(y_test,y_predict,labels=[2,4],target_names=["良性","恶性"]))

###模型评估
#ROC曲线与AUC值
# 把输出的 2 4 转换为 0 或 1
y_test = np.where(y_test > 2, 1, 0)  # 大于2就变为1，否则变为0
print("AUC值:\n",roc_auc_score(y_test,y_predict))
