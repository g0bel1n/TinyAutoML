import logging
from typing import Any, Union, List

import numpy as np
import pandas as pd

from ..MetaModel import MetaModel

pd.options.mode.chained_assignment = None  # default='warn'


class BestModel(MetaModel):
    def __init__(
        self,
        comprehensiveSearch: bool = True,
        parameterTuning: bool = True,
        metrics: str = "accuracy",
        nSplits: int = 10,
    ):
        self.best_estimator_name: str
        self.best_estimator: Any
        self.best_estimator_index: int

        # Pool of estimators
        super().__init__(comprehensiveSearch, parameterTuning, metrics, nSplits)
        self.scores = pd.DataFrame()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> MetaModel:

        super().fit(X, y, **kwargs)

        # Getting the best estimator according to the metric mean
        (
            best_score,
            self.best_estimator_name,
            self.best_estimator,
        ) = self.estimatorPool.get_best(X, y)

        logging.info(
            "The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
                self.best_estimator_name, best_score
            )
        )

        return self

    # Overloading sklearn BaseEstimator methods to use the best estimator
    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], **kwargs
    ) -> Union[pd.Series, np.ndarray, List[np.ndarray]]:

        try:
            pred = self.best_estimator.predict(X, **kwargs)
        except AttributeError as e:
            raise AttributeError from e
        return pred

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray], **kwargs
    ) -> Union[pd.Series, np.ndarray, List[np.ndarray]]:
        return self.best_estimator.predict_proba(X, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def __repr__(self, **kwargs):
        return "Best Model"
