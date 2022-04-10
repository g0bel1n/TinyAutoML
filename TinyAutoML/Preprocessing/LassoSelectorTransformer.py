import logging
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectKBest

from .LassoFeatureSelection import FeatureSelection


class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression
    """

    def __init__(self):

        self.selectedFeaturesNames = list[str]
        self.preSelectionSize = 50

    def __preselect(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Preselection for datasets with many features
        preselector = SelectKBest(k=self.preSelectionSize)
        preselector.fit(X, y)
        cols = preselector.get_support(indices=True).tolist()
        X = X.copy()
        return X.iloc[:, cols]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        X = self.__preselect(X, y) if (X.shape[1]) > self.preSelectionSize else X

        featureSelection = FeatureSelection()
        featureSelection.fit(X, y)

        self.selectedFeaturesNames = featureSelection.getSelectedFeaturesNames()
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        return X.loc[:,self.selectedFeaturesNames]
