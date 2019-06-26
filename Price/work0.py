import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  Ridge
from sklearn.model_selection import cross_val_score
import  PlotLearningCurve

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head())
print(df_test.info())
y_train = df_train.pop('SalePrice')

all_df = pd.concat((df_train, df_test), axis=0)
total = all_df.isnull().sum().sort_values(ascending=False)
percent = (all_df.isnull().sum()/ len(all_df)).sort_values(ascending=False)
miss_data = pd.concat([total, percent], axis=1, keys=['total','percent'])
all_df = all_df.drop(miss_data[miss_data['percent'] > 0.4].index, axis=1)

garage_obj=['GarageType','GarageFinish','GarageQual','GarageCond']
for garage in garage_obj:
   all_df[garage].fillna('missing', inplace=True)
all_df['GarageYrBlt'].fillna(1900., inplace=True)

all_df['MasVnrType'].fillna('missing', inplace=True)
all_df['MasVnrArea'].fillna(0, inplace=True)

plt.figure(figsize=(16, 6))
plt.plot(all_df['Id'], all_df['LotFrontage'])
plt.show()

all_df['LotFrontage'].fillna(all_df['LotFrontage'].mean(), inplace=True)
all_dummies_df = pd.get_dummies(all_df)
mean_col = all_dummies_df.mean()
all_dummies_df.fillna(mean_col, inplace=True)

scaler = StandardScaler()
scaler.fit(all_dummies_df.iloc[:, 1:])
all_dummies_df = scaler.transform(all_dummies_df.iloc[:, 1:])
print(all_dummies_df)


plt = plot_learning_curve()
