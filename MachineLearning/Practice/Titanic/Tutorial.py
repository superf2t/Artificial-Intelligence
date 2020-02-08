# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#####load data#####
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
print(data_train.head(10))
print(data_train.info())

#####1.EDA#####


#####2.Feature Engineering#####
##preprocess##
#Encoder
data_train["Sex"] = data_train["Sex"].apply(lambda x: 1 if x == "male" else 0)
data_test["Sex"] = data_test["Sex"].apply(lambda x: 1 if x == "male" else 0)

#Missing Values
data_train["Age"] = data_train.fillna(data_train["Age"].median())
data_test["Age"] = data_test.fillna(data_test["Age"].median())

#Feature Selection
feature = ["Age", "Sex"]


#####3.Model Selection#####
dt = DecisionTreeClassifier()
dt.fit(data_train[feature], data_train["Survived"])


######4.Model Ensembling#####


#####save result#####
predict_data = dt.predict(data_test[feature])
submission = pd.DataFrame({
    "PassengerId": data_test["PassengerId"],
    "Survived": predict_data
})
submission.to_csv("result.csv")

