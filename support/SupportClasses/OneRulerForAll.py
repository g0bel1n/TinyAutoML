from datetime import datetime

import pandas as pd
import logging
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB

from ..constants.GLOBAL_PARAMS import WINDOW
from ..constants.gsp import estimators_params


class OneRulerForAll(BaseEstimator):
    """
    Meta estimator that trains other estimator to find the best
    """

    def __init__(self, grid_search: bool, n_splits=10, ruler=None):

        if ruler is None:
            self.ruler = RandomForestClassifier()
            self.ruler_name = 'rfc'
        else:
            self.ruler = ruler
        self.estimators = [("rfc", RandomForestClassifier()),
                           ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                           ('Gaussian Naive Bayes', GaussianNB()),
                           ('LDA', LinearDiscriminantAnalysis()),
                           ]
        self.n_splits = n_splits
        self.grid_search = grid_search

    def fit(self, X: pd.DataFrame, y: pd.Series):

        logging.info("Training models...")

        y_train = y[WINDOW:]

        assert len(y[y == 1]) / len(y) <= 0.7, "The target is unbalanced"

        if X.index.dtype != datetime:
            cv = StratifiedKFold(n_splits=self.n_splits)
        else:
            cv = TimeSeriesSplit(n_splits=self.n_splits)

        if self.grid_search:
            for estimator in self.estimators:
                logging.info("\t----> {0}".format(estimator[0]))
                if estimator[0] in estimators_params:
                    clf = RandomizedSearchCV(estimator=estimator[1],
                                             param_distributions=estimators_params[estimator[0]], scoring='accuracy',
                                             n_jobs=-1, cv=cv)
                    clf.fit(X, y_train)

                    estimator[1].set_params(**clf.best_params_)

                estimator[1].fit(X, y_train)

            interm_ds = pd.DataFrame({estimator[0]: estimator[1].predict(X) for estimator in self.estimators})

        else:
            interm_ds = pd.DataFrame(
                {estimator[0]: estimator[1].fit(X, y_train).predict(X) for estimator in self.estimators})

        # self.final_estimator = GridSearchCV(estimator=xgb.XGBClassifier(use_label_encoder=False),
        #                                           param_grid=estimators_params['xgb'],
        #                                           scoring='accuracy',
        #                                           n_jobs=-1, cv=cv, refit=True, verbose=0)

        if self.ruler_name in estimators_params:
            clf = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                     param_distributions=estimators_params[self.ruler_name], scoring='accuracy',
                                     n_jobs=-1, cv=cv, refit=True)
            clf.fit(interm_ds, y_train)

        try : self.ruler.set_params(**clf.best_params_)
        except : pass

        self.ruler.fit(interm_ds, y_train)


        return self

    def predict(self, X: pd.Series) -> pd.Series:
        interm_ds = pd.DataFrame({estimator[0]: estimator[1].predict(X) for estimator in self.estimators})
        return self.ruler.predict(interm_ds)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        interm_ds = pd.DataFrame({estimator[0]: estimator[1].predict(X) for estimator in self.estimators})
        return self.ruler.predict_proba(interm_ds)

    def transform(self, X: pd.Series):
        return X
