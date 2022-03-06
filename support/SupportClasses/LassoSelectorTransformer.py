import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

from .FeatureSelection import FeatureSelection
from ..constants.GLOBAL_PARAMS import WINDOW


def lasso_thread(X: pd.DataFrame, y: pd.Series, i: int, tab_coeff: list, l_s: np.ndarray) -> None:
    lasso = LogisticRegression(C=l_s[i], fit_intercept=True, penalty='l1', solver='saga', max_iter=4000, verbose=0)
    lasso.fit(X, y)
    tab_coeff[i, :] = lasso.coef_[0]


class LassoSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression
    """

    def __init__(self, preSelectionSize=50):
        self.selectedFeaturesNames = [str]
        self.__preSelectionSize = preSelectionSize

    def __preselect(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        preselector = SelectKBest(k=self.__preSelectionSize)
        preselector.fit(X, y)
        cols = preselector.get_support(indices=True)
        X = X.copy()
        return X.iloc[:, cols]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerMixin:

        y = y[WINDOW:]  # ToDO correct that
        X = self.__preselect(X, y) if (X.shape[1]) > 50 else X
        featureSelection = FeatureSelection(X, y)
        featureSelection.doFeatureSelection()
        self.selectedFeaturesNames = featureSelection.getSelectedFeaturesNames()

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        return X[self.selectedFeaturesNames]
