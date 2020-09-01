import warnings
warnings.filterwarnings('ignore')
import pandas as pd

train = pd.read_csv('cs-training.csv')
test = pd.read_csv('cs-test.csv')

#处理缺失值
test['NumberOfDependents'].fillna(test['NumberOfDependents'].mode(), inplace=True)
train['NumberOfDependents'].fillna(train['NumberOfDependents'].mode(), inplace=True)
test['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean(), inplace=True)

#分箱处理
train.loc[(train['age'] >= 18) & (train['age'] < 40), 'age_range'] = 1
train.loc[(train['age'] >= 40) & (train['age'] < 60), 'age_range'] = 2
train.loc[(train['age'] >= 60) & (train['age'] < 80), 'age_range'] = 3
train.loc[(train['age'] >= 80), 'age_range'] = 4
train.loc[(train['age'] < 18), 'age_range'] = 4

test.loc[(test['age'] >= 18) & (test['age'] < 40), 'age_range'] = 1
test.loc[(test['age'] >= 40) & (test['age'] < 60), 'age_range'] = 2
test.loc[(test['age'] >= 60) & (test['age'] < 80), 'age_range'] = 3
test.loc[(test['age'] >= 80), 'age_range'] = 4
test.loc[(test['age'] < 18), 'age_range'] = 4


train.to_csv('cs-training0.csv', index=False)
test.to_csv('cs-test0.csv', index=False)
