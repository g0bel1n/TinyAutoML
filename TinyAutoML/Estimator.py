import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, roc_curve

from support.MyTools import buildMetaPipeline
from support.SupportClasses.MetaModel import MetaModel
from support.SupportClasses.DemocraticModel import DemocraticModel
from support.SupportClasses.OneRulerForAll import OneRulerForAll

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MetaModel_names = ["metamodel", "MetaModel", "Metamodel"]
ORFA_names = ["ORFA", "orfa", "OneRulerForAll", "onerulerforall"]
DemocraticModel_names = ["democraticmodel", "democratic", "Democratic", "DemocraticModel", "voting", "hardvoting", "Voting", "HardVoting"]

class MetaPipeline(BaseEstimator):

    def __getModel(self, modelName: str) -> tuple[str:BaseEstimator]:
        if modelName in MetaModel_names:
            return "Meta Model", MetaModel(grid_search=self.gridSearch, metrics=self.metrics)
        elif modelName in ORFA_names:
            return "ORFA", OneRulerForAll(gridSearch=self.gridSearch, ruler=self.ruler, metrics=self.metrics)
        elif modelName in DemocraticModel_names:
            return "Democratic Model", DemocraticModel(gridSearch=self.gridSearch, metrics=self.metrics, voting = self.voting)

    def __init__(self, model='orfa', gridSearch=True, ruler=None, verbose=True, metrics='accuracy', voting='soft'):
        assert model in MetaModel_names + ORFA_names + DemocraticModel_names, 'model not available'
        self.ruler = ruler #By default, it is a RandomForestClassifier, see class OneRulerForAll
        self.model = model
        self.gridSearch = gridSearch
        self.pipe = None
        self.verbose = verbose
        self.metrics = metrics
        self.voting = voting

        self.bottleNeckModel = self.__getModel(model)

        # To shut the logs
        if not verbose: logging.basicConfig(level=logging.CRITICAL)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:

        # some of the MetaPipeline steps requires information on the data, therefore we have to initialize it here
        self.pipe = buildMetaPipeline(X, self.bottleNeckModel)

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
        if self.bottleNeckModel[0] == 'ORFA':
            return 'scores are not available for ORFA model'
        else:
            return self.pipe.named_steps[self.bottleNeckModel[0]].scores

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
