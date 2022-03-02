import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from ..constants.GLOBAL_PARAMS import WINDOW


class NonStationarityCorrector(BaseEstimator, TransformerMixin):
    """
    Correct columns that don't seem stationary according to Augmented Dicker Fuller statistical stationarity test
    """

    def __init__(self, cols_to_pass=None):
        # If the user wants some columns to remain intact, they can be passed as arguments
        if cols_to_pass is None:
            cols_to_pass = []
        self.cols_to_pass = cols_to_pass
        self.cols_to_correct = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        for col in X.columns:
            if col not in self.cols_to_pass:
                test_results = adfuller(X[col].values)[0]
                if test_results < 0:
                    self.cols_to_correct.append(col)
                else:
                    self.cols_to_pass.append(col)
        assert len(self.cols_to_pass) + len(self.cols_to_correct) == len(X.columns), "Column(s) were lost in process"

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        logging.info("Correcting non-stationarity on the dataset...")
        if X.shape[0]>WINDOW:
            for col in self.cols_to_correct:
                # Depending on the window size, std can be null.
                # In that situation it can be replaced by the last non null value
                X[col] = (X[col] - X[col].rolling(window=WINDOW - 2).mean()) / X[col].rolling(window=WINDOW - 2).std(
                    skipna=True).replace(to_replace=0., method='ffill')

            if self.cols_to_pass: X[self.cols_to_pass] = StandardScaler().fit_transform(X[self.cols_to_pass])

            return X[WINDOW:]

        else:
            X[X.columns] = StandardScaler().fit_transform(X[X.columns])
            assert type(X) == pd.DataFrame, 'type error'
            return X
