import pandas as pd

from typing import Optional, Union
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

from .EstimatorsPool import EstimatorPool
from .EstimatorsPoolCV import EstimatorPoolCV

pd.options.mode.chained_assignment = None  # default='warn'


class MetaModel(ABC, BaseEstimator):

    comprehensiveSearch : bool
    is_fitted : bool
    estimatorPool : Union[EstimatorPool, EstimatorPoolCV]

    @abstractmethod
    def __init__(self,comprehensiveSearch: bool = True, parameterTuning: bool = True, metrics: str = 'accuracy', nSplits: int=10, ruler: Optional[BaseEstimator] =None):
        pass
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        pass

    # Overriding sklearn BaseEstimator methods
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_pool(self) -> Union[EstimatorPoolCV, EstimatorPool]:
        if self.estimatorPool is None : raise AttributeError('No pool was fitted or set')
        else : return self.estimatorPool

    def set_pool(self, fitted_pool: Union[EstimatorPoolCV, EstimatorPool] ):
        assert fitted_pool.is_fitted, "You must provide a fitted pool"
        self.estimatorPool = fitted_pool

