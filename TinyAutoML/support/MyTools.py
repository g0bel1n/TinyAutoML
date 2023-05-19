import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union
from pandas.api.types import is_datetime64_any_dtype
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold


def extract_score_params(
    cv_results: dict, n_splits: int, k_last_splits=5
) -> tuple[np.float64, dict]:
    """
    :param k_last_splits: k last splits to consider for averaging scores
    :param cv_results: output of parameterTuningCV.cv_results_
    :param n_splits: number of splits in the cross_validation
    :return: best score on the last k splits, and the params
    """
    if n_splits < k_last_splits:
        logging.warning(
            "There is more k_last_splits to include than the number of total splits."
            " k_last_splits will be set to {0}".format(n_splits)
        )

    idx_best_config = np.argmin(cv_results["rank_test_score"])
    score = np.mean(
        [
            cv_results["split{0}_test_score".format(k)][idx_best_config]
            for k in range(n_splits, n_splits - k_last_splits)
        ]
    )[0]
    params = cv_results["params"][idx_best_config]
    return score, params


def isIndexedByTime(X: pd.DataFrame) -> bool:
    """
    Checks if DataFrame index is datetime-like.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame whose index is to be checked.

    Returns
    -------
    bool
        True if index is datetime-like, False otherwise.
    """
    return is_datetime64_any_dtype(X.index)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_is_fitted

def get_df_scaler(scaler):
    class DataFrameScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = scaler
            self.columns = None  # Array to store input DataFrame column names

        def fit(self, X, y=None):
            # Handle the case when X is a DataFrame
            if isinstance(X, pd.DataFrame):
                self.columns = X.columns
            self.scaler.fit(X, y)
            return self

        def transform(self, X, y=None):
            check_is_fitted(self.scaler)
            X_scaled = self.scaler.transform(X)
            index = X.index if isinstance(X, pd.DataFrame) else None
            if self.columns is not None and isinstance(X, pd.DataFrame):
                X_scaled = pd.DataFrame(X_scaled, columns=self.columns, index=index)
            return X_scaled
    return DataFrameScaler

# Usage:
standard_scaler_df = get_df_scaler(StandardScaler())
minmax_scaler_df = get_df_scaler(MinMaxScaler())


def getAdaptedCrossVal(
    X: pd.DataFrame, nSplits: int
) -> Union[TimeSeriesSplit, StratifiedKFold]:
    if isIndexedByTime(X):
        # Pour des données temporelles, on préféra découper linéairement dans le temps
        return TimeSeriesSplit(n_splits=nSplits)

    else:
        # Si les données ne sont pas temporelles, on applique un découpage stratifié en prenant des
        return StratifiedKFold(n_splits=nSplits)


def checkClassBalance(y: pd.Series):
    if len(y[y == 1]) / len(y) > 0.7:
        logging.warning("Unbalanced Target")
        raise (ValueError)
    return True
