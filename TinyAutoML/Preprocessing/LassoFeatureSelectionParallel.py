from multiprocessing import Pool, cpu_count

from typing import Union, List
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from .PenalizationGrid import PenalizationGrid

pd.options.mode.chained_assignment = None  # default='warn'


class FeatureSelectionParallel:
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
        self.X: Union[pd.DataFrame, pd.Series]
        self.y: pd.Series

    def featureSelectionStep(self, penalizationCoeff: float):
        lasso = LogisticRegression(
            C=penalizationCoeff,
            fit_intercept=True,
            penalty="l1",
            solver="saga",
            max_iter=4000,
            verbose=0,
        )
        lasso.fit(self.X, self.y)

        return lasso.coef_[0]

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: pd.Series):
        self.X = X
        self.y = y
        with Pool(cpu_count() - 1) as pool:
            self.regressorsCoeffsValues = pool.map(
                self.featureSelectionStep, self.penalizationGrid.values
            )

        indexForObjNbOfFeatures = min(
            range(len(self.regressorsCoeffsValues)),
            key=lambda index: np.abs(
                sum(self.regressorsCoeffsValues[index] != 0) - self.nbFeatureToSelect
            ),
        )

        self.selectedFeaturesNames = X.columns[
            self.regressorsCoeffsValues[indexForObjNbOfFeatures] != 0
        ].values.tolist()

    def getSelectedFeaturesNames(self) -> List[str]:
        return self.selectedFeaturesNames
