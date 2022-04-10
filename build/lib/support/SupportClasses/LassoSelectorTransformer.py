import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectKBest

from .FeatureSelection import FeatureSelection


class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression
    """

    def __init__(self, preSelectionSize=50):
        self.selectedFeaturesNames = [str]
        self.__preSelectionSize = preSelectionSize

    def __preselect(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Preselection for datasets with many features
        preselector = SelectKBest(k=self.__preSelectionSize)
        preselector.fit(X, y)
        cols = preselector.get_support(indices=True)
        X = X.copy()
        return X.iloc[:, cols]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        X = self.__preselect(X, y) if (X.shape[1]) > self.__preSelectionSize else X

        featureSelection = FeatureSelection(X, y)
        featureSelection.doFeatureSelection()

        self.selectedFeaturesNames = featureSelection.getSelectedFeaturesNames()

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        return X[self.selectedFeaturesNames]
