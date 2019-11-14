#predict Boston housing price

import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
from PlotLearningCurve import *

def dataProcessing(df):
    field_cut = {
        'crim': [0, 10, 20, 100],
        'zn': [-1, 5, 18, 20, 40, 80, 86, 100],
        'indus': [-1, 7, 15, 23, 40],
        'nox': [0, 0.51, 0.6, 0.7, 0.8, 1],
        'rm': [0, 4, 5, 6, 7, 8, 9],
        'age': [0, 60, 80, 100],
        'dis': [0, 2, 6, 14],
        'rad': [0, 5, 10, 25],
        'tax': [0, 200, 400, 500, 800],
        'ptratio': [0, 14, 20, 23],
        'black': [0, 100, 350, 450],
        'lstat': [0, 5, 10, 20, 40]
    }
    cut_df = pd.DataFrame()
    for field in field_cut.keys():
        cut_series = pd.cut(df[field], field_cut[field], right=True)
        onehot_df = pd.get_dummies(cut_series, prefix=field)
        cut_df = pd.concat([cut_df, onehot_df], axis=1)
    neww_df = pd.concat([df, cut_df], axis=1)
    return neww_df

df = pd.read_csv("./train.csv")
print(df.head())

field_df = df.iloc[:, 1:-1]
print(field_df.head())
feature_df = dataProcessing(field_df)
print(feature_df.head())

X = feature_df
y = df['medv'].values
print(X.shape)
X = X[y != 50]
y = y[y != 50]
print(X.shape)

#{'learning_rate': 0.09, 'n_estimators': 200, 'max_depth': 4}
xgb_mdoel = XGBRegressor(learning_rate=0.09,n_estimators=200, max_depth=3, nthread=7)
cv_split = ShuffleSplit(n_splits=5, train_size=0.7, test_size=0.2)
grid_param = dict(
    max_depth=[4, 5, 6, 7],
    learning_rate=np.linspace(0.03, 0.3, 10),
    n_estimators=[100, 200]
)
# grid_model = GridSearchCV(xgb_mdoel, grid_param, cv=cv_split, scoring='neg_mean_squared_log_error')
grid_model = xgb_mdoel

plt = plot_learning_curve(grid_model, 'Curve', X.values, y, cv=cv_split)
plt.show()

grid_model.fit(X.values, y)
# print(grid_model.best_params_)
# print('rmse:', (grid_model.best_score_) ** 0.5)



predict_df = pd.read_csv("./test.csv")
predict_X = dataProcessing(predict_df.iloc[:, 1:]).values
predict_y = grid_model.predict(predict_X)
save_df = pd.DataFrame({'ID' : predict_df.ID, 'medv' : predict_y})
save_df.to_csv("result.csv", index=False)







