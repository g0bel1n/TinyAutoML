import logging

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from typing import Union

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_curve
from sklearn.pipeline import Pipeline

from .builders import buildMetaPipeline
from .Models import BestModel
from .Models import DemocraticModel
from .Models import OneRulerForAll


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MetaPipeline(BaseEstimator):
    #Wrapper

    def __init__(self, model: Union[BestModel,DemocraticModel,OneRulerForAll], verbose: bool=True):
        self.model = model
        self.pipe : Pipeline
        self.verbose = verbose
        # To shut the logs
        if not verbose: logging.basicConfig(level=logging.CRITICAL)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:

        # some of the MetaPipeline steps requires information on the data, therefore we have to initialize it here
        self.pipe = buildMetaPipeline(X, self.model)

        self.pipe.fit(X, y)

        return self

    #Overloading BaseEstimator methods
    def predict(self, X: pd.DataFrame):
        return self.pipe.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.pipe.predict_proba(X)

    def transform(self, X: pd.DataFrame, y=None):
        return self.pipe.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.pipe.fit(X, y).transform(X)

    def get_scores(self, X : pd.DataFrame, y: pd.Series):
            return self.pipe.named_steps[self.model.__repr__()].estimatorPool.get_scores(self.pipe.transform(X),y)

    def classification_report(self, X: pd.DataFrame, y: pd.Series):
        #Return sklearn classification report
        y_pred = self.pipe.predict(X)
        print(classification_report(y, y_pred))

    def roc_curve(self, X: pd.DataFrame, y: pd.Series):

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


