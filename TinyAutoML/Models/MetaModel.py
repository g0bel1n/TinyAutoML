from abc import ABC, abstractmethod
from typing import Optional, Union
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from ..support.MyTools import checkClassBalance, getAdaptedCrossVal

from .EstimatorPools.EstimatorPool import EstimatorPool
from .EstimatorPools.EstimatorPoolCV import EstimatorPoolCV

pd.options.mode.chained_assignment = None  # default='warn'


class MetaModel(ABC, BaseEstimator, ClassifierMixin):

    is_fitted: bool

    def __init__(
        self,
        comprehensiveSearch: bool = True,
        parameterTuning: bool = True,
        metrics: str = "accuracy",
        nSplits: int = 10,
    ):

        self.comprehensiveSearch = comprehensiveSearch
        self.parameterTuning = parameterTuning
        self.metrics = metrics
        self.nSplits = nSplits
        self.estimatorPool: Union[EstimatorPool, EstimatorPoolCV] = None  # type: ignore

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):

        checkClassBalance(y)

        logging.info("Training models...")

        cv = getAdaptedCrossVal(X, self.nSplits)

        if self.estimatorPool is None:
            self.estimatorPool = (
                EstimatorPoolCV() if self.comprehensiveSearch else EstimatorPool()
            )
            # Training the pool
            if self.parameterTuning:
                self.estimatorPool.fitWithParameterTuning(
                    X, y, cv, self.metrics, **kwargs
                )
            else:
                self.estimatorPool.fit(X, y, **kwargs)

    # Overriding sklearn BaseEstimator methods
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.Series, np.ndarray]:
        pass

    @abstractmethod
    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        pass

    @abstractmethod
    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        pass

    def get_pool(self) -> Union[EstimatorPoolCV, EstimatorPool]:
        if self.estimatorPool is None:
            raise AttributeError("No pool was fitted or set")
        else:
            return self.estimatorPool

    def set_pool(self, fitted_pool: Union[EstimatorPoolCV, Pipeline]):
        assert (
            type(fitted_pool) is EstimatorPoolCV and fitted_pool.is_fitted
        ), "You must provide a fitted pool"
        self.estimatorPool = fitted_pool
