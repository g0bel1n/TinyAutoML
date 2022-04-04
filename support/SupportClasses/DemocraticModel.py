import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .EstimatorsPool import EstimatorPool
from ..MyTools import getAdaptedCrossVal, checkClassBalance
from ..constants.gsp import estimators_params


class DemocraticModel(BaseEstimator):
    """
    The Democratic bottleneck estimator makes the trained models vote to decide the output.
    Classes are assumed to be in alphabetical order
    """

    def __init__(self, gridSearch: bool, metrics: str, nSplits=10):
        self.estimatorPool = EstimatorPool()
        self.nSplits = nSplits
        self.gridSearch = gridSearch
        self.metrics = metrics
        self.nEstimators = len(self.estimatorPool)
        

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:

        checkClassBalance(y)
        logging.info("Training models...")
        cv = getAdaptedCrossVal(X, self.nSplits)

        #Training the pool
        if self.gridSearch:
            self.estimatorPool.fitWithGridSearch(X,y,cv,self.metrics)
        else : self.estimatorPool.fit(X,y)

        return self

    # Overriding sklearn BaseEstimator methods
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        estimatorsPoolOutputs = self.estimatorPool.predict(X)
        estimator_names = estimatorsPoolOutputs.columns
        classes = np.sort(np.unique(estimatorsPoolOutputs.values))
        
        for c in classes:
            estimatorsPoolOutputs[f"{c}_proportion"] = (estimatorsPoolOutputs[estimator_names] == c).sum(axis=1) / self.nEstimators
        
        return estimatorsPoolOutputs[[f"{c}_proportion" for c in classes]].values

    def transform(self, X: pd.Series):
        return X