import threading

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from support.SupportClasses.PenalizationGrid import PenalizationGrid


class FeatureSelection:

    def __init__(self, X: pd.DataFrame, y: pd.Series, batchSize=10, nbFeatureToSelect=15):

        self.penalizationGrid = PenalizationGrid()
        self.regressorsCoeffsValues = []
        assert self.penalizationGrid.size() % batchSize == 0, 'The batch size must divide the size of the ' \
                                                              'penalization coefficients grid. '
        self.batchSize = batchSize
        self.nbFeatureToSelect = nbFeatureToSelect
        self.X, self.y = X, y
        self.selectedFeaturesNames = None

    def __featureSelectionStep(self, penalizationCoeff: float) -> None:
        lasso = LogisticRegression(C=penalizationCoeff, fit_intercept=True, penalty='l1', solver='saga', max_iter=4000,
                                   verbose=0)
        lasso.fit(self.X, self.y)
        self.regressorsCoeffsValues.append(lasso.coef_[0])

    def __featureSelectionBatchStep(self, penalizationPartialGrid: list):
        threads = [None] * self.batchSize

        for threadIndex, penalizationCoeff in enumerate(penalizationPartialGrid):
            threads[threadIndex] = threading.Thread(target=self.__featureSelectionStep, args=[penalizationCoeff])
            threads[threadIndex].start()

        for thread in threads:
            thread.join()

    def doFeatureSelection(self):

        while not self.penalizationGrid.isEmpty():
            penalizationPartialGrid = self.penalizationGrid.getNextKCoeffs(self.batchSize)
            self.__featureSelectionBatchStep(penalizationPartialGrid)

    def getSelectedFeaturesNames(self):
        indexForObjNbOfFeatures = min(range(len(self.regressorsCoeffsValues)), key=lambda index: np.abs(
            sum(self.regressorsCoeffsValues[index] != 0) - self.nbFeatureToSelect))
        self.selectedFeaturesNames = self.X.columns[self.regressorsCoeffsValues[indexForObjNbOfFeatures] != 0]
        return self.selectedFeaturesNames
