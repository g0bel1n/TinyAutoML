import logging
from typing import Union, List
from numpy import ndarray
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.base import TransformerMixin, BaseEstimator

from .LassoFeatureSelectionParallel import FeatureSelectionParallel

pd.options.mode.chained_assignment = None  # default='warn'


class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression.

    Parameters
    ----------
    preSelectionSize : int, optional
        Number of features to preselect. If the number of features in the input DataFrame is less than this, no preselection is performed.

    Attributes
    ----------
    selectedFeaturesNames : list of str
        Names of the selected features.
    """

    def __init__(self, preSelectionSize=50, k=15):
        self.selectedFeaturesNames = []
        self.preSelectionSize = preSelectionSize
        self.k = k
        self.skipPreSelection = False
        self.skipSelection = False

    def __preselect(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Preselect features using a SelectKBest.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series
            Target values.

        Returns
        -------
        X : pd.DataFrame
            DataFrame with preselected features.
        """
        self.skipPreSelection = X.shape[1] < self.preSelectionSize
        self.skipSelection = X.shape[1] < self.k
        
        if self.skipPreSelection:
            #logging.info("Number of features less than preSelectionSize, skipping pre-selection.")
            return X
        
        preselector = SelectKBest(k=self.preSelectionSize)
        preselector.fit(X, y)
        support = preselector.get_support(indices=True)
        cols = X.columns[support] if support is not None else X.columns
        return X.loc[:, cols]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:
        """
        Fit the feature selector to the data.

        If the number of features in the input DataFrame is greater than the pre-selection size, a pre-selection step is performed before fitting the feature selector. The names of the selected features are stored for use in the transform method.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series
            Target values.

        Returns
        -------
        self : TransformerMixin
        """
        if self.skipSelection:
            #logging.info("Number of features less than K, skipping feature selection.")
            self.selectedFeaturesNames = X.columns.tolist()
            return self

        X_ = self.__preselect(X, y)
        
        featureSelection = FeatureSelectionParallel(nbFeatureToSelect=self.k)
        featureSelection.fit(X_, y)

        self.selectedFeaturesNames = featureSelection.getSelectedFeaturesNames()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the input DataFrame using the selected features.

        The columns of the input DataFrame are filtered using the names of the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : None
            Not used. Included for compatibility with scikit-learn Transformer API.

        Returns
        -------
        X : pd.DataFrame
            DataFrame with selected features.
        """
        X = X.copy()
        return X.loc[:, self.selectedFeaturesNames]
