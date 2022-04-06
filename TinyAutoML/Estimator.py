import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_curve

from .builders import buildMetaPipeline


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MetaPipeline(BaseEstimator):

    def __init__(self, model: BaseEstimator, verbose=True):
        self.model = model
        self.pipe = None
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

    def get_scores(self):
        if self.model.__repr__() == 'ORFA':
            return 'scores are not available for ORFA model'
        else:
            return self.pipe.named_steps[self.model.__repr__()].scores

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


if __name__ == '__main__':
    MetaPipeline()
