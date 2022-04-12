import logging
import copy
from typing import Optional

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .EstimatorPool import EstimatorPool
from  ..support.MyTools import getAdaptedCrossVal, checkClassBalance
from  ..constants.gsp import estimators_params


class OneRulerForAll(BaseEstimator):
    """
    The OneRulerForAll bottleneck estimator uses stacking techniques to leverage the strengths of estimators in the pool.
    We first train a pool of estimator, then another 'ruler' estimator is trained to decide which estimator in
    the pool might be right, given the pool outputs
    """

    def __init__(self, parameterTuning: bool = True, metrics: str = 'accuracy', nSplits: int=10, ruler: Optional[BaseEstimator] =None):

        if ruler is None:
            self.ruler = RandomForestClassifier()
            self.rulerName = 'random forest classifier'
        else:
            self.ruler = ruler

        self.estimatorPool = EstimatorPool()
        self.nSplits = nSplits
        self.parameterTuning = parameterTuning
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.Series, pool = None) -> BaseEstimator:

        checkClassBalance(y)

        logging.info("Training models...")

        cv = getAdaptedCrossVal(X, self.nSplits)
        if pool is not None:
            self.estimatorPool = pool
        else:
        # Training the pool
            if self.parameterTuning:
                self.estimatorPool.fitWithparameterTuning(X, y, cv, self.metrics)
            else:
                self.estimatorPool.fit(X, y)

        estimatorsPoolOutputs = self.estimatorPool.predict(X)

        # Training the ruler, with parameterTuning if possible
        if self.rulerName in estimators_params:
            clf = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                     param_distributions=estimators_params[self.rulerName], scoring=self.metrics,
                                     n_jobs=-2, cv=cv)
            clf.fit(estimatorsPoolOutputs, y)
            self.ruler.set_params(**clf.best_params_)
        self.ruler.fit(estimatorsPoolOutputs, y)

        return self


    # Overriding sklearn BaseEstimator methods
    def predict(self, X: pd.DataFrame) -> pd.Series:
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        return self.ruler.predict(estimatorsPoolOutputs)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        return self.ruler.predict_proba(estimatorsPoolOutputs)

    def transform(self, X: pd.Series):
        return X

    def __repr__(self, **kwargs):
        return 'ORFA'
