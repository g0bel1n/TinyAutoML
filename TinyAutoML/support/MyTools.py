import logging
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold


def extract_score_params(cv_results: dict, n_splits: int, k_last_splits=5) -> tuple[np.float64, dict]:
    """
    :param k_last_splits: k last splits to consider for averaging scores
    :param cv_results: output of parameterTuningCV.cv_results_
    :param n_splits: number of splits in the cross_validation
    :return: best score on the last k splits, and the params
    """
    if n_splits < k_last_splits:
        logging.warning(
            'There is more k_last_splits to include than the number of total splits.'
            ' k_last_splits will be set to {0}'.format(
                n_splits))

    idx_best_config = np.argmin(cv_results['rank_test_score'])
    score = np.mean([cv_results['split{0}_test_score'.format(k)][idx_best_config] for k in
                     range(n_splits, n_splits - k_last_splits)])[0]
    params = cv_results['params'][idx_best_config]
    return score, params


def isIndexedByTime(X: pd.DataFrame) -> bool:
    return X.index.dtype in [datetime, 'datetime64[ns]']


def getAdaptedCrossVal(X: pd.DataFrame, nSplits: int) -> Union[TimeSeriesSplit, StratifiedKFold]:
    if isIndexedByTime(X):
        # Pour des données temporelles, on préféra découper linéairement dans le temps
        return TimeSeriesSplit(n_splits=nSplits)

    else:
        # Si les données ne sont pas temporelles, on applique un découpage stratifié en prenant des
        return StratifiedKFold(n_splits=nSplits)


def checkClassBalance(y: pd.Series):
    if len(y[y == 1]) / len(y) > 0.7:
        logging.warning('Unbalanced Target')
        raise(ValueError)
    return True
