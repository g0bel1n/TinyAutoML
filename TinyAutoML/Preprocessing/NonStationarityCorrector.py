import pandas as pd
import logging
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from ..constants.GLOBAL_PARAMS import WINDOW
from ..support.MyTools import isIndexedByTime

pd.options.mode.chained_assignment = None  # default='warn'


class NonStationarityCorrector(BaseEstimator, TransformerMixin):
    """
    Correct columns that don't seem stationary according to Augmented Dicker Fuller statistical stationarity test.

    Parameters
    ----------
    colsToKeepIntact : list, optional
        List of columns that should not be corrected even if they are found non-stationary.

    Attributes
    ----------
    colsToCorrect : list
        List of columns that need to be corrected for stationarity.
    """

    def __init__(self, colsToKeepIntact=None):
        if colsToKeepIntact is None:
            colsToKeepIntact = []
        self.colsToKeepIntact = colsToKeepIntact
        self.colsToCorrect = []
        self.indexedByTime = False

    def fit(self, X: pd.DataFrame, y=None) -> TransformerMixin:
        """
        Identify the columns in the DataFrame that need stationarity correction.

        The Augmented Dickey-Fuller test is applied to each numeric column in the DataFrame. Columns with p-values above 0.05 are considered non-stationary and are marked for correction. This method also checks whether the DataFrame index is of type datetime and logs a warning if it is not.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series
            Target values.

        Returns
        -------
        self : TransformerMixin
        """
        
        self.indexedByTime = isIndexedByTime(X)
        # Check if the DataFrame is time-indexed
        if not self.indexedByTime:
            logging.info("Index is not datetime. Stationarity correction was skipped.")
            return self

        # Identify columns to correct
        for col in X.columns:
            if col not in self.colsToKeepIntact and is_numeric_dtype(X[col]):
                p_val = adfuller(X[col].values)[1]
                if p_val > 0.05:  # 5% test
                    self.colsToCorrect.append(col)
                else:
                    self.colsToKeepIntact.append(col)

        assert len(self.colsToKeepIntact) + len(self.colsToCorrect) == len(
            X.columns
        ), "Column(s) were lost in process"
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply stationarity correction to the identified columns in the DataFrame.

        For each column marked for correction, the rolling mean and standard deviation are calculated and used to correct the column values. If the DataFrame has fewer rows than the window size, all columns in the DataFrame are standardized instead. This method also standardizes columns that are marked to be kept intact.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : None
            Not used. Included for compatibility with scikit-learn Transformer API.

        Returns
        -------
        X : pd.DataFrame
            DataFrame with corrected columns.
        """
        print(X)
        if not self.indexedByTime:
            return X
        
        X = X.copy()
        logging.debug("Correcting non-stationarity on the dataset...")

        if X.shape[0] > WINDOW:
            for col in self.colsToCorrect:
                X[col].iloc[WINDOW:] = (
                    (X[col] - X[col].rolling(window=WINDOW).mean())
                    / X[col]
                    .rolling(window=WINDOW)
                    .std(skipna=True)
                    .replace(to_replace=0.0, method="ffill")
                ).iloc[WINDOW:]

            if self.colsToKeepIntact:
                X[self.colsToKeepIntact] = StandardScaler().fit_transform(
                    X[self.colsToKeepIntact]
                )
        else:
            X[X.columns] = StandardScaler().fit_transform(X[X.columns])
            assert isinstance(X, pd.DataFrame), "type error"

        return X
