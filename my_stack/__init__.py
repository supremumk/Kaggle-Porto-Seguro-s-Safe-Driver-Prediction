import sys
sys.path.insert(0, "/home/jzkay/xgboost/python-package")
import pandas as pd
import numpy as np
from sklearn.utils import resample

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import *
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier  # https://github.com/fukatani/rgf_python
from sklearn.tree import DecisionTreeClassifier


# train = train.drop(['target', 'id'], axis=1)
# test = test.drop(['id'], axis=1)

# col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
# train = train.drop(col_to_drop, axis=1)
# test = test.drop(col_to_drop, axis=1)

# remove duplicate variables
# remove duplicated columns
# remove = []
# c = train.columns
# for i in range(len(c)-1):
#     v = train[c[i]].values
#     for j in range(i+1,len(c)):
#         if np.array_equal(v,train[c[j]].values):
#             remove.append(c[j])
#
# train.drop(remove, axis=1, inplace=True)
# test.drop(remove, axis=1, inplace=True)

# train = train.replace(-1, np.nan)
# test = test.replace(-1, np.nan)
# train[np.where(np.isnan(train))]=-1
# test[np.where(np.isnan(test))]=-1


# cat_features = [a for a in train.columns if a.endswith('cat')]
#
# for column in cat_features:
#     temp = pd.get_dummies(pd.Series(train[column]))
#     train = pd.concat([train, temp], axis=1)
#     train = train.drop([column], axis=1)
#
# for column in cat_features:
#     temp = pd.get_dummies(pd.Series(test[column]))
#     test = pd.concat([test, temp], axis=1)
#     test = test.drop([column], axis=1)
# Columns -> binary decoded.
def replace_at_index(TUP, ix, val):
    lst = list(TUP)
    lst[ix] = val
    return tuple(lst)

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        # S_test = np.zeros((T.shape[0], len(self.base_models)))
        for j, (train_idx, test_idx) in enumerate(folds):
            if j == 0:
                for i, clf in enumerate(self.base_models):
            # S_test_i = np.zeros((T.shape[0], self.n_splits))



                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_holdout = X[test_idx]
                    #                y_holdout = y[test_idx]

                    print ("Fit %s fold %d" % (str(clf).split('(')[0], j + 1))
                    clf.fit(X_train, y_train)
                    #                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
                    #                print("    cross_score: %.5f" % (cross_score.mean()))
                    y_pred = clf.predict_proba(X_holdout)

                    S_train[test_idx, i] = y_pred[:,1]

                    self.base_models = replace_at_index(self.base_models, i, clf)


                self.stacker.fit(np.column_stack((np.ones(S_train[test_idx, :].shape[0]), S_train[test_idx, :])),
                             y[test_idx])
            else:
                pass
                # S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            # S_test[:, i] = S_test_i.mean(axis=1)


        # results = cross_val_score(self.stacker, S_train[test_idx, :], y[test_idx], cv=1, scoring='roc_auc')
        # print("Stacker score: %.5f" % (results.mean()))


        # res = self.stacker.predict_proba(np.column_stack((np.ones(S_test.shape[0]),S_test)))[:, 1]
        return None

    def predict_proba(self, T):
        T=np.array(T)
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test[:, i] = clf.predict_proba(T)[:,1]

        res = self.stacker.predict_proba(np.column_stack((np.ones(S_test.shape[0]),S_test)))
        return res



def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

# Funcitons from olivier's kernel
# https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


