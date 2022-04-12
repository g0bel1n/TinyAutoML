import logging

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .EstimatorsPoolCV import EstimatorPoolCV

from ..support.MyTools import  getAdaptedCrossVal, checkClassBalance


class BestModelCV(BaseEstimator):

    def __init__(self, parameterTuning: bool =True, metrics: str ='accuracy', n_splits: int =10):
        self.best_estimator_name : str
        self.best_estimator_ : Pipeline
        self.best_estimator_index : int

        # Pool of estimators
        self.estimatorPoolCV = EstimatorPoolCV()
        self.scores = pd.DataFrame()
        self.n_splits = n_splits
        self.parameterTuning = parameterTuning
        self.metrics = metrics


    def fit(self, X: pd.DataFrame, y: pd.Series, ) -> BaseEstimator:


        logging.info("Training models")

        # Pour détecter une distribution déséquilibrée...
        checkClassBalance(y)
        # Récupération d'un split de CV adapté selon l'indexage du set
        cv = getAdaptedCrossVal(X, self.n_splits)

        if self.parameterTuning:
            self.estimatorPoolCV.fitWithparameterTuning(X,y,cv,'accuracy')
        else:
            self.estimatorPoolCV.fit(X, y)

        # Getting the best estimator according to the metric mean
        best_score , self.best_estimator_name, self.best_estimator = self.estimatorPoolCV.get_best(X,y)

        logging.info("The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
            self.best_estimator_name, best_score))

        return self

    # Overloading sklearn BaseEstimator methods to use the best estimator
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.best_estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return self.best_estimator.predict_proba(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def __repr__(self, **kwargs):
        return 'Meta Model CV'
