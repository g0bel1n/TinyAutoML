from typing import Any, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from ...constants.gsp import estimators_params

pd.options.mode.chained_assignment = None  # default='warn'


class EstimatorPool:
    def __init__(self):

        self.estimatorsList = [
            ("random forest classifier", RandomForestClassifier()),
            ("Logistic Regression", LogisticRegression(fit_intercept=True)),
            ("Gaussian Naive Bayes", GaussianNB()),
            ("LDA", LinearDiscriminantAnalysis()),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
        ]

        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, **kwargs
    ) -> list[tuple[str, ClassifierMixin]]:
        for estimator in self.estimatorsList:
            estimator[1].fit(X, y, **kwargs)
        self.is_fitted = True
        return self.estimatorsList

    def fitWithParameterTuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[TimeSeriesSplit, StratifiedKFold],
        metric,
        **kwargs,
    ) -> list[tuple[str, ClassifierMixin]]:

        for estimator in self.estimatorsList:
            if estimator[0] in estimators_params:
                grid = estimators_params[estimator[0]]
                clf = RandomizedSearchCV(
                    estimator=estimator[1],
                    param_distributions=grid,
                    scoring=metric,
                    n_jobs=-2,
                    cv=cv,
                    n_iter=min(10, len(grid)),
                )
                clf.fit(X, y, **kwargs)

                estimator[1].set_params(**clf.best_params_)

            estimator[1].fit(X, y, **kwargs)

        self.is_fitted = True
        return self.estimatorsList

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                estimator[0]: estimator[1].predict(X, **kwargs)
                for estimator in self.estimatorsList
            }
        )

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> ndarray:
        return np.array(
            [
                estimator[1].predict_proba(X, **kwargs)
                for estimator in self.estimatorsList
            ]
        )

    def get_best(
        self, X: Union[pd.DataFrame, np.ndarray], y: pd.Series, **kwargs
    ) -> tuple[float, str, Any]:

        scores = [
            accuracy_score(estimator[1].predict(X, **kwargs), y)
            for estimator in self.estimatorsList
        ]
        return float(np.max(scores)), *self.estimatorsList[np.argmax(scores)]

    def get_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> list[tuple[str, float]]:
        return [
            (estimator_name, accuracy_score(estimator.predict(X, **kwargs), y))
            for estimator_name, estimator in self.estimatorsList
        ]

    def __len__(self, **kwargs):
        return len(self.estimatorsList)
