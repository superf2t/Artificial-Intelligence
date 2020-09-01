from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
columns = ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']
train_age = train[columns]
test_age = test[columns]
train_known_age = train_age[train_age.Age.notnull()].values
train_unknown_age = train_age[train_age.Age.isnull()].values
test_unknown_age = test_age[test_age.Age.isnull()].values

#y即目标年龄
y = train_known_age[:, 0]
#X即特征属性值
X = train_known_age[:, 1:]

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
rfr.fit(X, y)

# 用得到的模型进行未知年龄结果预测
predicted_age = rfr.predict(train_unknown_age[:, 1:])
# 用得到的预测结果填补原缺失数据
train.loc[(train.Age.isnull()), 'Age'] = predicted_age
predicted_age = rfr.predict(test_unknown_age[:, 1:])
test.loc[test.Age.isnull(), 'Age'] = predicted_age

#类别用众数填充
train['Embarked'].fillna(train['Embarked'].mode(), inplace=True)

#数值用中位数填充
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# #set Carbin type
train.loc[(train.Cabin.notnull()), 'Cabin'] = "Yes"
train.loc[(train.Cabin.isnull()), 'Cabin'] = "No"
test.loc[(test.Cabin.notnull()), 'Cabin'] = "Yes"
test.loc[(test.Cabin.isnull()), 'Cabin'] = "No"

#get_dummies
dummies_carbin = pd.get_dummies(train['Cabin'], prefix='Cabin')
dummies_embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
dummies_sex = pd.get_dummies(train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')
train = pd.concat([train, dummies_carbin, dummies_embarked, dummies_sex, dummies_Pclass],
                  axis=1)
train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

dummies_carbin = pd.get_dummies(test['Cabin'], prefix='Cabin')
dummies_embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
dummies_sex = pd.get_dummies(test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')
test = pd.concat([test, dummies_carbin, dummies_embarked, dummies_sex, dummies_Pclass],
                  axis=1)
test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = StandardScaler()
age_scale_param = scaler.fit(train[['Age']])
train['Age_scaled'] = scaler.fit_transform(train[['Age']], age_scale_param)
test['Age_scaled'] = scaler.fit_transform(test[['Age']], age_scale_param)

fare_scale_param = scaler.fit(train[['Fare']])
train['Fare_scaled'] = scaler.fit_transform(train[['Fare']], fare_scale_param)
test['Fare_scaled'] = scaler.fit_transform(test[['Fare']], fare_scale_param)

test.to_csv("./test0.csv", index=False)
train.to_csv("./train0.csv", index=False)

