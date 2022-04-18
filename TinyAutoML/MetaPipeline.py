import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, roc_curve
from sklearn.pipeline import Pipeline

from .builders import buildMetaPipeline
from .Models import EstimatorPool, EstimatorPoolCV, MetaModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pd.options.mode.chained_assignment = None  # default='warn'


class MetaPipeline(BaseEstimator, ClassifierMixin):
    # Wrapper

    def __init__(self, model: MetaModel, verbose: bool = True):
        self.model = model
        self.estimator: Optional[Union[Pipeline, MetaModel]] = None
        self.verbose = verbose
        # To shut the logs
        if not verbose:
            logging.basicConfig(level=logging.CRITICAL)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        pool: Optional[Union[EstimatorPool, EstimatorPoolCV]] = None,
    ) -> BaseEstimator:
        if (type(y) is np.ndarray and len(y.shape) != 1) or (
            type(y) is pd.Series and y.shape[1] != 1
        ):
            raise ValueError("The target is not a vector")
        # some of the MetaPipeline steps requires information on the data, therefore we have to initialize it here
        self.estimator = (
            self.model
            if self.model.comprehensiveSearch
            else buildMetaPipeline(X, self.model)
        )
        if pool is not None:
            self.__set_pool(pool)

        self.estimator.fit(X, y)

        return self

    def get_pool(self) -> Optional[Union[EstimatorPoolCV, EstimatorPool]]:
        try:
            if type(self.estimator) is MetaModel:
                return self.estimator.get_pool()
            elif type(self.estimator) is Pipeline:
                return self.estimator.named_steps[self.model.__repr__()].get_pool()
        except AttributeError as e:
            raise AttributeError("Please, fit the estimator beforehand") from e

    def __set_pool(self, fitted_pool: Union[EstimatorPool, EstimatorPoolCV]):
        if type(self.estimator) is MetaModel:
            self.estimator.set_pool(fitted_pool)
        elif type(self.estimator) is Pipeline:
            self.estimator.named_steps[self.model.__repr__()].set_pool(fitted_pool)

    # Overloading BaseEstimator methods
    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        if self.estimator is None:
            raise ValueError("The estimator has not been fitted beforehand")
        return self.estimator.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if self.estimator is None:
            raise ValueError("The estimator has not been fitted beforehand")
        return self.estimator.predict_proba(X)

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if self.estimator is None:
            raise ValueError("The estimator has not been fitted beforehand")
        return self.estimator.transform(X)

    def get_scores(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ):
        if (type(y) is np.ndarray and len(y.shape) != 1) or (
            type(y) is pd.Series and y.shape[1] != 1
        ):
            raise ValueError("The target is not a vector")

        if type(self.estimator) is MetaModel:
            return self.estimator.estimatorPool.get_scores(
                self.estimator.transform(X), y
            )
        elif type(self.estimator) is Pipeline:
            return self.estimator.named_steps[
                self.model.__repr__()
            ].estimatorPool.get_scores(self.estimator.transform(X), y)

    def classification_report(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ):
        # Return sklearn classification report
        if self.estimator is None:
            raise ValueError("The estimator has not been fitted beforehand")
        if (type(y) is np.ndarray and len(y.shape) != 1) or (
            type(y) is pd.Series and y.shape[1] != 1
        ):
            raise ValueError("The target is not a vector")

        y_pred = self.estimator.predict(X)
        print(classification_report(y, y_pred))

    def roc_curve(self, X: pd.DataFrame, y: pd.Series):

        if self.estimator is None:
            raise ValueError("The estimator has not been fitted beforehand")
        if (type(y) is np.ndarray and len(y.shape) != 1) or (
            type(y) is pd.Series and y.shape[1] != 1
        ):
            raise ValueError("The target is not a vector")

        ns_probs = [0 for _ in range(len(y))]

        # predict probabilities
        lr_probs = self.estimator.predict_proba(X)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
        plt.plot(lr_fpr, lr_tpr, marker=".", label="Logistic")
        # axis labels
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
