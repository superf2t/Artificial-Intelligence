import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

y_train = df_train.pop('SalePrice')

df_all = pd.concat((df_train, df_test), axis=0)
total = df_all.isnull().sum().sort_values(ascending=False)  #每列缺失数量
percent = (df_all.isnull().sum()/len(df_all)).sort_values(ascending=False) #每列缺失率
miss_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])

df_all = df_all.drop(miss_data[miss_data['percent'] > 0.4].index, axis=1) #去除了percent>0.4的列
df_all['GarageYrBlt'].fillna(1900., inplace=True)

train_labels = df_train.pop('SalePrice')

#merge train and test
features = pd.concat([df_train, df_test], keys=['train', 'test'])

#delete some feature
#check missing data
#drop some missing data
#fill the rest missing data
#Converting categorical data to dummies

# MSSubClass as str
df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)

df_all['date_parsed'] = pd.to_datetime(df_all['date'],format="%m/%d/%y")


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


est = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.to_csv( 'titanic_pred.csv' , index = False )

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)