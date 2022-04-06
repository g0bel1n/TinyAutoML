import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer

from TinyAutoML.Preprocessing.LassoSelectorTransformer import LassoSelectorTransformer
from TinyAutoML.Preprocessing.NonStationarityCorrector import NonStationarityCorrector


def buildColumnTransformer(X: pd.DataFrame) -> ColumnTransformer:

    # Select numerical and categorical feature to be able to apply different transformers

    numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

    if X.index.dtype == 'datetime64[ns]':
        numerical_process = Pipeline([('NonStationarityCorrector', NonStationarityCorrector()),
                                      ('MinMaxScaler', MinMaxScaler(feature_range=[-1, 1]))])
    else:
        numerical_process = Pipeline([('StandardScaler', StandardScaler()),
                                      ('MinMaxScaler', MinMaxScaler(feature_range=[-1, 1]))])


    transformer = [('Categorical', OneHotEncoder(), categorical_ix),
                   ("Numerical", numerical_process, numerical_ix)]

    return ColumnTransformer(transformers=transformer)


def buildMetaPipeline(X: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    cols = X.columns
    columnTransformer = buildColumnTransformer(X)

    if len(cols) > 15:
        return Pipeline(
            [('Preprocessing', columnTransformer),
             ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
             ("Lasso Selector", LassoSelectorTransformer()),
             (estimator.__repr__(), estimator)])
    else:
        return Pipeline(
            [('Preprocessing', columnTransformer),
             ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
             (estimator.__repr__(), estimator)])