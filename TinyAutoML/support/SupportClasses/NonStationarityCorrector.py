import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from typing import Any
from ..constants.GLOBAL_PARAMS import WINDOW


class NonStationarityCorrector(BaseEstimator, TransformerMixin):
    """
    Correct columns that don't seem stationary according to Augmented Dicker Fuller statistical stationarity test

    """

    def __init__(self, cols_to_pass=None):
        if cols_to_pass is None:
            cols_to_pass = []
        self.cols_to_pass = cols_to_pass
        self.cols_to_correct = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        i, j, k = 0, 0, 0
        for col in X.columns:
            if col not in self.cols_to_pass:
                test_results = adfuller(X[col].values)
                if test_results[0] < 0:
                    self.cols_to_correct.append(col)
                    j += 1
                else:
                    self.cols_to_pass.append(col)
                    k += 1
                i += 1
        assert len(self.cols_to_pass) + len(self.cols_to_correct) == len(X.columns), "Column(s) were lost in process"

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        logging.info("Correcting non-stationarity on the dataset...")
        for col in self.cols_to_correct:
            X[col] = (X[col] - X[col].rolling(window=WINDOW - 2).mean()) / X[col].rolling(window=WINDOW - 2).std(
                skipna=True).replace(to_replace=0., method='ffill')

        sc = StandardScaler()
        if self.cols_to_pass: X[self.cols_to_pass] = sc.fit_transform(X[self.cols_to_pass])

        assert type(X) == pd.DataFrame, 'type error'

        return X[WINDOW:]
