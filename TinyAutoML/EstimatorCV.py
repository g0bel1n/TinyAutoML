from typing import Union
import logging

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_curve

from .Models import BestModelCV
from .Models import DemocraticModelCV
from .Models import OneRulerForAllCV


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MetaPipelineCV(BaseEstimator):
    #Wrapper

    def __init__(self, model: Union[BestModelCV,DemocraticModelCV,OneRulerForAllCV], verbose: bool=True):
        self.model = model
        self.verbose = verbose
        # To shut the logs
        if not verbose: logging.basicConfig(level=logging.CRITICAL)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:

        # some of the MetaPipeline steps requires information on the data, therefore we have to initialize it here
        self.model.fit(X,y)

        return self

    #Overloading BaseEstimator methods
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def transform(self, X: pd.DataFrame, y=None):
        return self.model.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.model.fit(X, y).transform(X)

    def get_scores(self,X: pd.DataFrame,y: pd.Series) -> list[tuple[str, float]]:
        return self.model.estimatorPoolCV.get_scores(self.transform(X),y)

    def classification_report(self, X: pd.DataFrame, y: pd.Series):
        #Return sklearn classification report
        y_pred = self.model.predict(X)
        print(classification_report(y, y_pred))

    def roc_curve(self, X: pd.DataFrame, y: pd.Series):

        ns_probs = [0 for _ in range(len(y))]

        # predict probabilities
        lr_probs = self.model.predict_proba(X)
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
    
    MetaPipelineCV(DemocraticModelCV())
