import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('cs-training.csv')

#EDA
#显示所有列
pd.set_option("display.max_columns", None)
#显示所有行
pd.set_option("display.max_rows", None)
print(data_train.head())
print(data_train.dtypes)
print(data_train.describe())
print(data_train.info())
#MonthlyIncome and NumberOfDependents have null

#看下数据有没有重复值
print("Duplicated: ", data_train.duplicated().sum())
#删除重复值
data_train = data_train.drop_duplicates()

#列名重命名
colnames={'SeriousDlqin2yrs':'Isdlq',
         'RevolvingUtilizationOfUnsecuredLines':'Revol',
         'NumberOfTime30-59DaysPastDueNotWorse':'Num30-59late',
         'NumberOfOpenCreditLinesAndLoans':'Numopen',
         'NumberOfTimes90DaysLate':'Num90late',
         'NumberRealEstateLoansOrLines':'Numestate',
         'NumberOfTime60-89DaysPastDueNotWorse':'Num60-89late',
         'NumberOfDependents':'Numdepend'}
data_train.rename(columns=colnames,inplace=True)

sns.countplot('Isdlq', data=data_train)
badNum = data_train.loc[data_train['Isdlq'] == 1, :].shape[0]
goodNum = data_train.loc[data_train['Isdlq'] == 0, :].shape[0]
print('好坏比：{0}%'.format(round(badNum*100/(goodNum+badNum), 2)))
#data imbalance

#Age数据分布情况
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(data_train['age'], ax=ax1)
sns.boxplot(y='age', data=data_train, ax=ax2)
#异常值情况
age_mean=data_train['age'].mean()
age_std=data_train['age'].std()
age_lowlimit = age_mean - 3 * age_std
age_uplimit = age_mean + 3 * age_std
print('异常值下限：', age_lowlimit, '异常值上限：', age_uplimit)

age_lowlimitd = data_train.loc[data_train['age'] < age_lowlimit, :]
age_uplimitd = data_train.loc[data_train['age'] > age_uplimit, :]
print('异常值下限比例：{0}%'.format(age_lowlimitd.shape[0]*100/data_train.shape[0]),
     '异常值上限比例：{0}%'.format(age_uplimitd.shape[0]*100/data_train.shape[0]))

data_age = data_train.loc[data_train['age'] > 0, ['age', 'Isdlq']]
data_age.loc[(data_age['age'] > 18) & (data_age['age'] < 40), 'age'] = 1
data_age.loc[(data_age['age'] >= 40) & (data_age['age'] < 60), 'age'] = 2
data_age.loc[(data_age['age'] >= 60) & (data_age['age'] < 80), 'age'] = 3
data_age.loc[(data_age['age'] >= 80), 'age'] = 4
age_Isdlq = data_age.groupby('age')['Isdlq'].sum()
age_total = data_age.groupby('age')['Isdlq'].count()
age_Isratio = age_Isdlq/age_total
age_Isratio.plot(kind='bar', figsize=(8, 6), color='#4682B4')
plt.show()