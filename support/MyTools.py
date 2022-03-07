import logging
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer

from support.SupportClasses.LassoSelectorTransformer import LassoSelectorTransformer
from support.SupportClasses.NonStationarityCorrector import NonStationarityCorrector


def extract_score_params(cv_results: dict, n_splits: int, k_last_splits=5) -> tuple[np.float64, dict]:
    """
    :param k_last_splits: k last splits to consider for averaging scores
    :param cv_results: output of GridSearchCV.cv_results_
    :param n_splits: number of splits in the cross_validation
    :return: best score on the last k splits, and the params
    """
    if n_splits < k_last_splits:
        logging.warning(
            'There is more k_last_splits to include than the number of total splits.'
            ' k_last_splits will be set to {0}'.format(
                n_splits))
        k_last_splits = n_splits

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
    if not len(y[y == 1]) / len(y) <= 0.7 :
        logging.warning('Unbalanced Target')
        raise(ValueError)
    return True


def buildColumnTransformer(X: pd.DataFrame) -> ColumnTransformer:

    # Select numerical and categorical feature to be able to apply different transformers

    numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

    numerical_process = Pipeline([('NonStationarityCorrector', NonStationarityCorrector()),
                                  ('MinMaxScaler', MinMaxScaler(feature_range=[-1, 1]))])

    transformer = [('Categorical', OneHotEncoder(), categorical_ix),
                   ("Numerical", numerical_process, numerical_ix)]

    return ColumnTransformer(transformers=transformer)


def buildMetaPipeline(X: pd.DataFrame, bottleNeckEstimator: tuple[str, BaseEstimator]) -> Pipeline:
    cols = X.columns
    columnTransformer = buildColumnTransformer(X)

    if len(cols) > 15:
        return Pipeline(
            [('Preprocessing', columnTransformer),
             ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
             ("Lasso Selector", LassoSelectorTransformer()),
             bottleNeckEstimator])
    else:
        return Pipeline(
            [('Preprocessing', columnTransformer),
             ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
             bottleNeckEstimator])
