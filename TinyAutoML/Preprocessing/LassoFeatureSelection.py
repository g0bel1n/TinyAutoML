import threading
from typing import Union
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from .PenalizationGrid import PenalizationGrid

pd.options.mode.chained_assignment = None  # default='warn'


class FeatureSelection:
    def __init__(self, batchSize=10, nbFeatureToSelect=15):

        self.penalizationGrid = PenalizationGrid()
        self.regressorsCoeffsValues = []
        assert self.penalizationGrid.size() % batchSize == 0, (
            "The batch size must divide the size of the "
            "penalization coefficients grid. "
        )
        self.batchSize = batchSize
        self.nbFeatureToSelect = nbFeatureToSelect
        self.selectedFeaturesNames = None

    def __featureSelectionStep(self, penalizationCoeff: float, X, y) -> None:
        lasso = LogisticRegression(
            C=penalizationCoeff,
            fit_intercept=True,
            penalty="l1",
            solver="saga",
            max_iter=4000,
            verbose=0,
        )
        lasso.fit(X, y)
        self.regressorsCoeffsValues.append(lasso.coef_[0])

    def __featureSelectionBatchStep(self, penalizationPartialGrid: list, X, y):
        threads = {}

        for threadIndex, penalizationCoeff in enumerate(penalizationPartialGrid):
            threads[threadIndex] = threading.Thread(
                target=self.__featureSelectionStep, args=[penalizationCoeff, X, y]
            )
            threads[threadIndex].start()

        for thread in threads.values():
            thread.join()

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: pd.Series):

        while not self.penalizationGrid.isEmpty():
            penalizationPartialGrid = self.penalizationGrid.getNextKCoeffs(
                self.batchSize
            )
            self.__featureSelectionBatchStep(penalizationPartialGrid, X, y)

        indexForObjNbOfFeatures = min(
            range(len(self.regressorsCoeffsValues)),
            key=lambda index: np.abs(
                sum(self.regressorsCoeffsValues[index] != 0) - self.nbFeatureToSelect
            ),
        )

        self.selectedFeaturesNames = X.columns[
            self.regressorsCoeffsValues[indexForObjNbOfFeatures] != 0
        ].values.tolist()

    def getSelectedFeaturesNames(self) -> list[str]:
        return self.selectedFeaturesNames
