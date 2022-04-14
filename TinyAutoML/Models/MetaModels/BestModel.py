import logging
import numpy as np
import pandas as pd

from typing import Any, Optional, Union

from ..MetaModel import MetaModel
from ..EstimatorPools.EstimatorPool import EstimatorPool
from ..EstimatorPools.EstimatorPoolCV import EstimatorPoolCV
from ...support.MyTools import getAdaptedCrossVal, checkClassBalance


pd.options.mode.chained_assignment = None  # default='warn'

class BestModel(MetaModel):

    def __init__(self, comprehensiveSearch: bool = True, parameterTuning: bool =True, metrics: str ='accuracy', n_splits: int =10):
        self.best_estimator_name : str
        self.best_estimator : Any
        self.best_estimator_index : int

        # Pool of estimators
        self.estimatorPool  : Optional[Union[EstimatorPoolCV, EstimatorPool]] = None
        self.scores = pd.DataFrame()
        self.n_splits = n_splits
        self.parameterTuning = parameterTuning
        self.metrics = metrics
        self.comprehensiveSearch = comprehensiveSearch


    def fit(self, X: pd.DataFrame, y: pd.Series, ) -> MetaModel:


        logging.info("Training models")

        # Pour détecter une distribution déséquilibrée...
        checkClassBalance(y)
        # Récupération d'un split de CV adapté selon l'indexage du set
        cv = getAdaptedCrossVal(X, self.n_splits)

        if self.estimatorPool is None :
            self .estimatorPool = EstimatorPoolCV() if self.comprehensiveSearch else EstimatorPool()
        # Training the pool
            if self.parameterTuning:

                self.estimatorPool.fitWithParameterTuning(X, y, cv, self.metrics)
            else:
       
                self.estimatorPool.fit(X, y)

        # Getting the best estimator according to the metric mean
        best_score , self.best_estimator_name, self.best_estimator = self.estimatorPool.get_best(X,y)

        logging.info("The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
            self.best_estimator_name, best_score))

        return self

    # Overloading sklearn BaseEstimator methods to use the best estimator
    def predict(self, X: Union[pd.DataFrame,np.ndarray]) -> Union[pd.Series,np.ndarray, list[np.ndarray]]:

        if hasattr(self.best_estimator, 'predict') is True : self.best_estimator.predict(X)
        try: pred = self.best_estimator.predict(X)
        except AttributeError as e:
            raise AttributeError from e
        return pred

    def predict_proba(self, X: Union[pd.DataFrame,np.ndarray]) -> Union[pd.Series,np.ndarray, list[np.ndarray]]:
        return self.best_estimator.predict_proba(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def __repr__(self, **kwargs):
        return 'Best Model'