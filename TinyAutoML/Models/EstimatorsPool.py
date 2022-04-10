from lib2to3.pytree import Base
from typing import Union, Tuple, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     TimeSeriesSplit)
from sklearn.naive_bayes import GaussianNB
from TinyAutoML.constants.gsp import estimators_params


class EstimatorPool(BaseEstimator):

    def __init__(self):

        self.estimatorsList = [("random forest classifier", RandomForestClassifier()),
                               ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                               ('Gaussian Naive Bayes', GaussianNB()),
                               ('LDA', LinearDiscriminantAnalysis()),
                               ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, BaseEstimator]]:
        for estimator in self.estimatorsList: estimator[1].fit(X, y)
        return self.estimatorsList

    def fitWithGridSearch(self, X: pd.DataFrame, y: pd.Series,
                          cv: Union[TimeSeriesSplit, StratifiedKFold],
                          metrics) -> list[tuple[str, BaseEstimator]]:

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
        
    def predict_proba(self, X: pd.DataFrame) -> ndarray:
        return np.array([estimator[1].predict_proba(X) for estimator in self.estimatorsList])

    def get_best(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, str, BaseEstimator]:

        scores = [accuracy_score(estimator[1].predict(X), y) for estimator in self.estimatorsList]
        return float(np.max(scores)), *self.estimatorsList[np.argmax(scores)]

    def get_scores(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]: 
        return [(estimator_name, accuracy_score(estimator.predict(X), y)) for estimator_name,estimator in self.estimatorsList]
        
    def __len__(self):
        return len(self.estimatorsList)
