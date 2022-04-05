from typing import Union

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from support.constants.gsp import estimators_params


class EstimatorPool(BaseEstimator):

    def __init__(self):

        self.estimatorsList = [("random forest classifier", RandomForestClassifier()),
                               ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                               ('Gaussian Naive Bayes', GaussianNB()),
                               ('LDA', LinearDiscriminantAnalysis()),
                               ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> [tuple[str, BaseEstimator]]:
        for estimator in self.estimatorsList: estimator[1].fit(X, y)
        return self.estimatorsList

    def fitWithGridSearch(self, X: pd.DataFrame, y: pd.Series,
                          cv: Union[TimeSeriesSplit, StratifiedKFold],
                          metrics) -> [tuple[str, BaseEstimator]]:

        for estimator in self.estimatorsList:
            if estimator[0] in estimators_params:
                clf = RandomizedSearchCV(estimator=estimator[1],
                                         param_distributions=estimators_params[estimator[0]], scoring=metrics,
                                         n_jobs=-1, cv=cv)
                clf.fit(X, y)

                estimator[1].set_params(**clf.best_params_)

            estimator[1].fit(X, y)

        return self.estimatorsList

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {estimator[0]: estimator[1].predict(X) for estimator in self.estimatorsList})
        
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.array([estimator[1].predict_proba(X) for estimator in self.estimatorsList])
        
    def __len__(self):
        return len(self.estimatorsList)
