import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from estimator.src.constants.GLOBAL_PARAMS import WINDOW
from estimator.src.SupportClasses.LassoSelector import LassoSelector
from estimator.src.SupportClasses.MetaModel import MetaModel
from estimator.src.SupportClasses.NonStationarityCorrector import NonStationarityCorrector



class MetaPipeline(BaseEstimator):

    def __init__(self):
        self.pipe = None

    def fit(self, X: pd.DataFrame, y: pd.Series, grid_search=True):

        cols = X.columns
        numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

        transformer = [('Categorical', OneHotEncoder(), categorical_ix),
                       ("Numerical", NonStationarityCorrector(), numerical_ix)]

        col_transformer = ColumnTransformer(transformers=transformer)

        if len(cols) > 15:
            self.pipe = Pipeline(
                [('Preprocessing', col_transformer),
                 ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                 ("Lasso Selector", LassoSelector()),
                 ("Meta Model", MetaModel(grid_search=grid_search))])
        else:
            self.pipe = Pipeline(
                [('Preprocessing', col_transformer),
                 ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                 ("Meta Model", MetaModel(grid_search=grid_search))])

        self.pipe.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame):
        return self.pipe.predict(X)

    def transform(self, X: pd.DataFrame, y=None):
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.pipe.fit(X, y).transform(X)

    def get_scores(self):
        return self.pipe.named_steps['Meta Model'].scores

    def classification_report(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.pipe.predict(X)
        print(classification_report(y[WINDOW:], y_pred))
