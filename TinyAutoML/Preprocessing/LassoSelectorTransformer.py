import logging
from typing import Union
from numpy import ndarray
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.base import TransformerMixin, BaseEstimator

from .LassoFeatureSelection import FeatureSelection

pd.options.mode.chained_assignment = None  # default='warn'


class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression
    """

    def __init__(self, preSelectionSize=50):

        self.selectedFeaturesNames = list[str]
        self.preSelectionSize = preSelectionSize

    def __preselect(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Union[pd.DataFrame, pd.Series]:
        # Preselection for datasets with many features
        preselector = SelectKBest(k=self.preSelectionSize)
        preselector.fit(X, y)
        support = preselector.get_support(indices=True)
        cols = support.tolist() if support is not None else X.columns
        X = X.copy()
        return X.iloc[:, cols]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        X_ = self.__preselect(X, y) if (X.shape[1]) > self.preSelectionSize else X

        featureSelection = FeatureSelection()
        featureSelection.fit(X_, y)

        self.selectedFeaturesNames = featureSelection.getSelectedFeaturesNames()
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()

        return X.loc[:, self.selectedFeaturesNames]
