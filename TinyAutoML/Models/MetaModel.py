from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .EstimatorPools.EstimatorPool import EstimatorPool
from .EstimatorPools.EstimatorPoolCV import EstimatorPoolCV

pd.options.mode.chained_assignment = None  # default='warn'


class MetaModel(ABC, BaseEstimator, ClassifierMixin):

    comprehensiveSearch: bool
    is_fitted: bool
    estimatorPool: Union[EstimatorPool, EstimatorPoolCV]

    @abstractmethod
    def __init__(
        self,
        comprehensiveSearch: bool = True,
        parameterTuning: bool = True,
        metrics: str = "accuracy",
        nSplits: int = 10,
    ):
        pass

    @abstractmethod
    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> ClassifierMixin:
        pass

    # Overriding sklearn BaseEstimator methods
    @abstractmethod
    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
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

    def set_pool(self, fitted_pool: Union[EstimatorPoolCV, EstimatorPool]):
        assert fitted_pool.is_fitted, "You must provide a fitted pool"
        self.estimatorPool = fitted_pool
