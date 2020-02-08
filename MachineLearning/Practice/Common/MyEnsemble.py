from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier
import copy
import numpy as np

class AverageEnsemble(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.__models = copy.deepcopy(models)

    def fit(self, X, y):
        for model in self.__models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.__models])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = None
        self.meta_model_ = None

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = copy.deepcopy(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = copy.deepcopy(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

def StackingClassifier(clfs, train_x, train_y, test_x):
    dataset_blend_train = np.zeros((train_x.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((test_x.shape[0], len(clfs)))

    '''5折stacking'''
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1).split(train_x, train_y)
    for j, clf in enumerate(clfs):
        '''依次训练各个单模型'''
        # print(j, clf)
        dataset_blend_test_j = np.zeros((test_x.shape[0], n_folds))
        for i, (train, test) in enumerate(skf):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            X_train, y_train, X_test, y_test = train_x[train], train_y[train], train_x[test], train_y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(test_x)[:, 1]
        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_blend_train, train_y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    return y_submission

def BlendingClassifier(clfs, train_x, train_y, test):
    # 切分训练数据集为d1,d2两部分
    X_d1, X_d2, y_d1, y_d2 = train_test_split(train_x, train_y, test_size=0.5, random_state=2017)
    dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
    dataset_d2 = np.zeros((test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        # 依次训练各个单模型
        clf.fit(X_d1, y_d1)
        y_submission = clf.predict_proba(X_d2)[:, 1]
        dataset_d1[:, j] = y_submission
        # 对于测试集，直接用这k个模型的预测值作为新的特征。
        dataset_d2[:, j] = clf.predict_proba(test)[:, 1]

    # 融合使用的模型
    clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
    clf.fit(dataset_d1, y_d2)
    y_submission = clf.predict_proba(dataset_d2)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    return y_submission
