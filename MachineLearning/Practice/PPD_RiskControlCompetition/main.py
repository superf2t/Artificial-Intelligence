import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# f_train1 = pd.read_csv('first_train1.csv',encoding='gbk')
# f_train2 = pd.read_csv('first_train2.csv',encoding='gbk')
# f_train3 = pd.read_csv('first_train3.csv',encoding='gbk')
# f_test1 = pd.read_csv('first_test1.csv',encoding='gb18030')
# f_test2 = pd.read_csv('first_test2.csv',encoding='gbk')
# f_test3 = pd.read_csv('first_test3.csv',encoding='gbk')
# #训练集和测试集合并
# f_train1['sample_status'] = 'train'
# f_test1['sample_status'] = 'test'
# df1 = pd.concat([f_train1, f_test1], axis=0).reset_index(drop=True)
# df2 = pd.concat([f_train2, f_test2], axis=0).reset_index(drop=True)
# df3 = pd.concat([f_train3, f_test3], axis=0).reset_index(drop=True)
# # 保存数据至本地
# df1.to_csv('data_input1.csv', encoding='gb18030', index=False)
# df2.to_csv('data_input2.csv', encoding='gb18030', index=False)
# df3.to_csv('data_input3.csv', encoding='gb18030', index=False)

# 导入 data_input 处理好的数据
df1 = pd.read_csv('first_train1.csv', encoding='gb18030')
print(df1.shape)
# 样本的好坏比
print(df1.target.value_counts())
print(df1.info())


# 借款成交时间的范围
import datetime as dt
df1['ListingInfo'] = pd.to_datetime(df1.ListingInfo)
df1['month'] = df1.ListingInfo.dt.strftime('%Y-%m')
plt.figure(figsize=(10, 4))
plt.title('借款成交量的时间趋势图')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
sns.countplot(data=df1.sort_values('month'), x='month')
plt.show()

# 根据月份计算每个月的违约率
month_group = df1.groupby('month')
time_bad_trend = pd.DataFrame()
time_bad_trend['total'] = month_group.target.count()
time_bad_trend['bad'] = month_group.target.sum()
time_bad_trend['bad_rate'] = time_bad_trend['bad']/time_bad_trend['total']
time_bad_trend = time_bad_trend.reset_index()
plt.figure(figsize=(12, 4))
plt.title('违约率的时间趋势图')
sns.pointplot(data=time_bad_trend, x='month', y='bad_rate', linestyles='-')
plt.show()

print(df1.info())
null_sum = df1.isnull().sum()
null_sum = null_sum[null_sum != 0]
null_sum_df = pd.DataFrame(null_sum, columns=['num'])
null_sum_df['ratio'] = null_sum_df['num'] / df1.shape[0]
null_sum_df.sort_values(by='ratio', ascending=False, inplace=True)
print(null_sum_df.head(10))
plt.figure(figsize=(15, 5))
sns.barplot(data=null_sum_df, y='ratio', x=null_sum_df.index)
plt.show()

# 检查数值型变量的缺失
# 原始数据中-1作为缺失的标识，将-1替换为np.nan
data1 = df1.drop(['ListingInfo', 'month'], axis=1)
data1 = data1.replace({-1: np.nan})
missing_columns = list(null_sum_df[null_sum_df['ratio'] > 0.8].index)
print('We will remove %d columns.' % len(missing_columns))

# 删除缺失值比例高于50%的列
data = df1.drop(columns = list(missing_columns))
data1.to_csv('C:/Users/Administrator/Desktop/魔镜杯数据/data1_clean.csv',encoding='gb18030',index=False)