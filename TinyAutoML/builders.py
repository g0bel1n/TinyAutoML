from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from .Preprocessing.LassoSelectorTransformer import LassoSelectorTransformer
from .Preprocessing.NonStationarityCorrector import NonStationarityCorrector
from .support.MyTools import isIndexedByTime


def buildColumnTransformer(X: Union[pd.DataFrame, np.ndarray]) -> ColumnTransformer:

    # Select numerical and categorical feature to be able to apply different transformers
    if type(X) is pd.DataFrame:
        numerical_ix = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_ix = X.select_dtypes(include=["object", "bool"]).columns

    elif type(X) is np.ndarray:
        numerical_ix = X.dtype(include=["int64", "float64"]).columns
        categorical_ix = X.dtype(include=["object", "bool"]).columns

    else:
        raise ValueError("X is not valid")

    if type(X) is pd.DataFrame and isIndexedByTime(X):

        numerical_process = Pipeline(
            [
                ("NonStationarityCorrector", NonStationarityCorrector()),
                ("MinMaxScaler", MinMaxScaler(feature_range=[-1, 1])),
            ]
        )
    else:
        numerical_process = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("MinMaxScaler", MinMaxScaler(feature_range=[-1, 1])),
            ]
        )

    transformer = [
        ("Categorical", OneHotEncoder(), categorical_ix),
        ("Numerical", numerical_process, numerical_ix),
    ]

    return ColumnTransformer(transformers=transformer)


def buildMetaPipeline(
    X: Union[pd.DataFrame, np.ndarray], estimator: BaseEstimator
) -> Pipeline:

    cols = (
        X.columns
        if type(X) is pd.DataFrame
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    columnTransformer = buildColumnTransformer(X)

    if len(cols) > 15:
        return Pipeline(
            [
                ("Preprocessing", columnTransformer),
                ("fc_tf", FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                ("Lasso Selector", LassoSelectorTransformer()),
                (estimator.__repr__(), estimator),
            ]
        )
    else:
        return Pipeline(
            [
                ("Preprocessing", columnTransformer),
                ("fc_tf", FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                (estimator.__repr__(), estimator),
            ]
        )
