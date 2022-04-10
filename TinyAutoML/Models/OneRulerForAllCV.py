import logging
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from TinyAutoML.Models.EstimatorsPoolCV import EstimatorPoolCV

from .EstimatorsPool import EstimatorPool
from TinyAutoML.support.MyTools import getAdaptedCrossVal, checkClassBalance
from TinyAutoML.constants.gsp import estimators_params


class OneRulerForAllCV(BaseEstimator):
    """
    The OneRulerForAll bottleneck estimator uses stacking techniques to leverage the strengths of estimators in the pool.
    We first train a pool of estimator, then another 'ruler' estimator is trained to decide which estimator in
    the pool might be right, given the pool outputs
    """

    def __init__(self, gridSearch: bool = True, metrics: str = 'accuracy', nSplits: int=10, ruler: Optional[BaseEstimator] =None):

        if ruler is None:
            self.ruler = RandomForestClassifier()
            self.rulerName = 'random forest classifier'
        else:
            self.ruler = ruler

        self.estimatorPoolCV = EstimatorPoolCV()
        self.nSplits = nSplits
        self.gridSearch = gridSearch
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:

        checkClassBalance(y)

        logging.info("Training models...")

        cv = getAdaptedCrossVal(X, self.nSplits)

        # Training the pool
        if self.gridSearch:
            self.estimatorPoolCV.fitWithGridSearch(X, y, cv, self.metrics)
        else:
            self.estimatorPoolCV.fit(X, y)

        estimatorsPoolOutputs = self.estimatorPoolCV.predict(X)

        self.ruler.fit(estimatorsPoolOutputs, y)

        return self

    # Overriding sklearn BaseEstimator methods
    def predict(self, X: pd.DataFrame) -> pd.Series:
        estimatorsPoolOutputs = self.estimatorPoolCV.predict(X)
        return self.ruler.predict(estimatorsPoolOutputs)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        estimatorsPoolOutputs = self.estimatorPoolCV.predict(X)
        return self.ruler.predict_proba(estimatorsPoolOutputs)

    def transform(self, X: pd.DataFrame):
        return X

    def __repr__(self, **kwargs):
        return 'ORFA CV'
