

import numpy as np
import pandas as pd

from typing import Union
from numpy import ndarray
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     TimeSeriesSplit)
from sklearn.naive_bayes import GaussianNB

from ..constants.gsp import estimators_params

pd.options.mode.chained_assignment = None  # default='warn'

class EstimatorPool():

    def __init__(self):

        self.estimatorsList = [("random forest classifier", RandomForestClassifier()),
                               ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                               ('Gaussian Naive Bayes', GaussianNB()),
                               ('LDA', LinearDiscriminantAnalysis()),
                               ('xgb', XGBClassifier(use_label_encoder=False))
                               ]
        
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, BaseEstimator]]:
        for estimator in self.estimatorsList: estimator[1].fit(X, y)
        self.is_fitted = True
        return self.estimatorsList

    def fitWithparameterTuning(self, X: pd.DataFrame, y: pd.Series,
                          cv: Union[TimeSeriesSplit, StratifiedKFold],
                          metrics) -> list[tuple[str, BaseEstimator]]:

        for estimator in self.estimatorsList:
            if estimator[0] in estimators_params:
                grid = estimators_params[estimator[0]]
                clf = RandomizedSearchCV(estimator=estimator[1],
                                         param_distributions=grid, scoring=metrics,
                                         n_jobs=-2, cv=cv, n_iter=min(10, len(grid)))
                clf.fit(X, y)

                estimator[1].set_params(**clf.best_params_)

            estimator[1].fit(X, y)

        self.is_fitted = True
        return self.estimatorsList

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {estimator[0]: estimator[1].predict(X) for estimator in self.estimatorsList})
        
    def predict_proba(self, X: pd.DataFrame) -> ndarray:
        return np.array([estimator[1].predict_proba(X) for estimator in self.estimatorsList])

    def get_best(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, str, BaseEstimator]:

        scores = [accuracy_score(estimator[1].predict(X), y) for estimator in self.estimatorsList]
        return float(np.max(scores)), *self.estimatorsList[np.argmax(scores)]

    def get_scores(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]: 
        return [(estimator_name, accuracy_score(estimator.predict(X), y)) for estimator_name,estimator in self.estimatorsList]
        
    def __len__(self):
        return len(self.estimatorsList)
