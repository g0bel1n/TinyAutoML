import logging
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from ..constants.GLOBAL_PARAMS import WINDOW

pd.options.mode.chained_assignment = None  # default='warn'


class NonStationarityCorrector(BaseEstimator, TransformerMixin):
    """
    Correct columns that don't seem stationary according to Augmented Dicker Fuller statistical stationarity test
    """

    def __init__(self, colsToKeepIntact=None):
        # If the user wants some columns to remain intact, they can be passed as arguments
        if colsToKeepIntact is None:
            colsToKeepIntact = []
        self.colsToKeepIntact = colsToKeepIntact
        self.colsToCorrect = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        """
        split the cols according to Augmented Dicker Fuller statistical stationarity test
        """
        for col in X.columns:
            if col not in self.colsToKeepIntact:
                p_val = adfuller(X[col].values)[1]
                if p_val > 0.05: #Test à 5%
                    self.colsToCorrect.append(col)
                else:
                    self.colsToKeepIntact.append(col)

        assert len(self.colsToKeepIntact) + len(self.colsToCorrect) == len(
            X.columns
        ), "Column(s) were lost in process"
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Apply the transformations
        X = X.copy()
        logging.debug("Correcting non-stationarity on the dataset...")
        if X.shape[0] > WINDOW:
            for col in self.colsToCorrect:
                # Depending on the window size, std can be null.
                # In that situation the actual value can be replaced by the last non null value
                # We also use loc[start+WINDOW:] in order to leave the WINDOW first rows intact. Otherwise, it would be nans
                X[col].iloc[WINDOW:] = (  # type: ignore
                    (X[col] - X[col].rolling(window=WINDOW).mean())
                    / X[col]
                    .rolling(window=WINDOW)
                    .std(skipna=True)
                    .replace(to_replace=0.0, method="ffill")
                ).iloc[
                    WINDOW:
                ]  # type: ignore

            if self.colsToKeepIntact:
                X[self.colsToKeepIntact] = StandardScaler().fit_transform(
                    X[self.colsToKeepIntact]
                )
        else:
            X[X.columns] = StandardScaler().fit_transform(X[X.columns])
            assert type(X) == pd.DataFrame, "type error"

        return X
