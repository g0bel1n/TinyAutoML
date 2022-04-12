from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.base import BaseEstimator


class MetaModel(ABC, BaseEstimator):

    @abstractmethod
    def __init__(self, parameterTuning: bool = True, metrics: str = 'accuracy', nSplits: int=10, ruler: Optional[BaseEstimator] =None):
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
    def transform(self, X: pd.DataFrame):
        pass
