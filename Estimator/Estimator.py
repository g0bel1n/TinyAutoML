import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler

from ..support.constants.GLOBAL_PARAMS import WINDOW
from ..support.SupportClasses.LassoSelector import LassoSelector
from ..support.SupportClasses.MetaModel import MetaModel
from ..support.SupportClasses.OneRulerForAll import OneRulerForAll as orfa
from ..support.SupportClasses.NonStationarityCorrector import NonStationarityCorrector


class MetaPipeline(BaseEstimator):

    def __init__(self,Model:str, grid_search=True):
        assert Model in ['metamodel', 'orfa'], 'model not available'
        if Model == 'metamodel':
            self.bottle_neck_estimator = ("Meta Model", MetaModel(grid_search=grid_search))
        else : self.bottle_neck_estimator = ("ORFA", orfa(grid_search=grid_search))

        self.pipe = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

        cols = X.columns
        numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

        numerical_process = Pipeline([('NonStationarityCorrector', NonStationarityCorrector()),
                                      ('MinMaxScaler', MinMaxScaler(feature_range=[-1, 1]))])

        transformer = [('Categorical', OneHotEncoder(), categorical_ix),
                       ("Numerical", numerical_process, numerical_ix)]

        col_transformer = ColumnTransformer(transformers=transformer)

        if len(cols) > 15:
            self.pipe = Pipeline(
                [('Preprocessing', col_transformer),
                 ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                 ("Lasso Selector", LassoSelector()),
                 self.bottle_neck_estimator])
        else:
            self.pipe = Pipeline(
                [('Preprocessing', col_transformer),
                 ('fc_tf', FunctionTransformer(lambda x: pd.DataFrame(x, columns=cols))),
                 self.bottle_neck_estimator])

        self.pipe.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame):
        return self.pipe.predict(X)

    def transform(self, X: pd.DataFrame, y=None):
        return self.pipe.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.pipe.fit(X, y).transform(X)

    def get_scores(self):
        return self.pipe.named_steps['Meta Model'].scores

    def classification_report(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.pipe.predict(X)
        print(classification_report(y[WINDOW:], y_pred))

    def roc_curve(self,X: pd.DataFrame, y:pd.Series):
        y = y[WINDOW:]
        ns_probs = [0 for _ in range(len(y))]

        # predict probabilities
        lr_probs = self.pipe.predict_proba(X)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()


if __name__ == '__main__':
    Pipeline()
