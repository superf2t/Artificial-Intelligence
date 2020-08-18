
def DataPreporcess(data_train):
    #####异常值处理

    #age异常值处理
    data_train = data_train[data_train['age'] > 0]

    #Num30-59late Num60-89late Num90late异常值处理
    data_train = data_train[data_train['Num30-59late'] < 90]
    data_train = data_train[data_train['Num60-89late'] < 90]
    data_train = data_train[data_train['Num90late'] < 90]

    #Numestate异常值处理
    data_train = data_train[data_train['Numestate'] < 50]


    #####缺失值处理
    # Numdepend缺失值处理
    data_train['Numdepend'] = data_train['NumDepend'].fillna(data_train['NumDepend'].mean(), inplace=True)

    # MonthlyIncome缺失值处理
    # 随机森林预测缺失值
    data_Forest=data_train.iloc[:, [5, 1, 2, 3, 4, 6, 7, 8, 9]]
    MonthlyIncome_isnull = data_Forest.loc[data_train['MonthlyIncome'].isnull(), :]
    MonthlyIncome_notnull=data_Forest.loc[data_train['MonthlyIncome'].notnull(), :]
    from sklearn.ensemble import RandomForestRegressor
    X = MonthlyIncome_notnull.iloc[:, 1:].values
    y = MonthlyIncome_notnull.iloc[:, 0].values
    regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=200, n_jobs=-1)
    regr.fit(X, y)
    MonthlyIncome_fillvalue = regr.predict(MonthlyIncome_isnull.iloc[:, 1:].values).round(0)

    # 填充MonthlyIncome缺失值
    data_train.loc[data_train['MonthlyIncome'].isnull(), 'MonthlyIncome'] = MonthlyIncome_fillvalue