import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from sklearn.naive_bayes import GaussianNB
from typing import Any

from ..constants.GLOBAL_PARAMS import WINDOW
from ..constants.gsp import estimators_params
from datetime import datetime
from sklearn.model_selection import GridSearchCV



class MetaModel(BaseEstimator):

    def __init__(self, grid_search: bool, n_splits=10):
        self.best_estimator = None
        self.best_estimator_index = None
        self.estimators = [("rcf", RandomForestClassifier()),
                           ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                           ('Gaussian Naive Bayes', GaussianNB()),
                           ('LDA', LinearDiscriminantAnalysis())]
        self.scores = pd.DataFrame()
        self.n_splits = n_splits
        self.grid_search = grid_search

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:

        print("Training models")

        y_train = y[WINDOW:]

        assert len(y[y == 1]) / len(y) <= 0.8, "The target is unbalanced"

        if (X.index.dtype != datetime):
            cv = KFold(n_splits=self.n_splits)
        else:
            cv = TimeSeriesSplit(n_splits=self.n_splits)

        dict_scores = {'mean': []}
        if self.grid_search:
            print('Training : ')
            for estimator in self.estimators:
                print("---->", estimator[0])
                if estimator[0] in estimators_params:
                    clf = GridSearchCV(estimator[1], param_grid=estimators_params[estimator[0]], scoring='accuracy',
                                       n_jobs=-1, cv=cv,refit=True)
                    clf.fit(X, y_train)
                    b_ix = np.argmax(clf.cv_results_['mean_test_score'])
                    dict_scores['mean'].append(clf.cv_results_['mean_test_score'][b_ix])
                else:
                    cvs = cross_val_score(estimator[1], X, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                    dict_scores['mean'].append(cvs.mean())

            self.scores = pd.DataFrame(dict_scores, index=[estimator[0] for estimator in self.estimators])

        else:
            self.scores = pd.DataFrame(
                data=[cross_val_score(estimator[1], X, y_train, cv=cv, scoring='accuracy', n_jobs=-1) for estimator in
                      self.estimators], index=[estimator[0] for estimator in self.estimators],
                columns=['Split {0}'.format(i) for i in range(self.n_splits)])

            self.scores['mean'] = self.scores.mean(axis=1)

        self.best_estimator_index = self.scores['mean'].argmax()

        print('\tDone.')
        print('Results : ')
        print("The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
            self.estimators[self.best_estimator_index][0], self.scores['mean'].iloc[self.best_estimator_index]))

        self.best_estimator = self.estimators[self.best_estimator_index][1].fit(X, y_train)

        return self

    def predict(self, X: pd.Series) -> pd.Series:
        return self.best_estimator.predict(X)
