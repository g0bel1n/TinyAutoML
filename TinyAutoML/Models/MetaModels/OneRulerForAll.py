from typing import Union

import pandas as pd
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from ..MetaModel import MetaModel

pd.options.mode.chained_assignment = None  # default='warn'


class OneRulerForAll(MetaModel):
    """
    The OneRulerForAll bottleneck estimator uses stacking techniques to leverage the strengths of estimators in the pool.
    We first train a pool of estimator, then another 'ruler' estimator is trained to decide which estimator in
    the pool might be right, given the pool outputs
    """

    def __init__(
        self,
        comprehensiveSearch: bool = True,
        parameterTuning: bool = True,
        metrics: str = "accuracy",
        nSplits: int = 10,
        ruler=None,
    ):
        super().__init__(comprehensiveSearch, parameterTuning, metrics, nSplits)
        self.ruler = RandomForestClassifier() if ruler is None else ruler

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MetaModel:

        super().fit(X, y)

        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        X_ruler = pd.concat((X, estimatorsPoolOutputs), axis=1)
        self.ruler.fit(X_ruler, y)

        return self

    # Overriding sklearn BaseEstimator methods
    def predict(self, X: pd.DataFrame) -> Union[pd.Series, ndarray, list[ndarray]]:
        assert (
            self.estimatorPool is not None and self.estimatorPool.is_fitted
        ), "Please fit the model before"
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        X_ruler = pd.concat((X, estimatorsPoolOutputs), axis=1)
        return self.ruler.predict(X_ruler)

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.Series, ndarray, list[ndarray]]:
        assert (
            self.estimatorPool is not None and self.estimatorPool.is_fitted
        ), "Please fit the model before"
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        X_ruler = pd.concat((X, estimatorsPoolOutputs), axis=1)
        return self.ruler.predict_proba(X_ruler)

    def transform(self, X: pd.DataFrame):
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        return pd.concat((X, estimatorsPoolOutputs), axis=1)

    def __repr__(self, **kwargs):
        return "ORFA"
