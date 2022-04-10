import logging
import threading
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from ..constants.GLOBAL_PARAMS import WINDOW


def lasso_thread(X: pd.DataFrame, y: pd.Series, i: int, tab_coeff: list, l_s: np.ndarray)->None:
    lasso = LogisticRegression(C=l_s[i], fit_intercept=True, penalty='l1', solver='saga', max_iter=4000, verbose=0)
    lasso.fit(X, y)
    tab_coeff[i, :] = lasso.coef_[0]


class LassoSelector(BaseEstimator, TransformerMixin):
    """
    Selects features according to their order of apparition when varying the shrinkage coefficient of LASSO regression
    """

    def __init__(self):
        self.selected_cols = [str]

    def fit(self, X: pd.DataFrame, y: pd.Series)->TransformerMixin:

        y

        if(X.shape[1])>50:
            # pre- feature selection
            selector = SelectKBest(k=50).fit(X, y)
            cols = selector.get_support(indices=True)
            X = X.copy()
            X = X.iloc[:, cols]

        coeffs_list = []
        l_s = np.linspace(0.3, 0.0001, 100)

        lasso = LogisticRegression(C=l_s[0], fit_intercept=True, penalty='l1', solver='saga', max_iter=4000)
        lasso.fit(X, y)
        coeffs_list.append(lasso.coef_[0])
        X = X[np.array(X.columns)[coeffs_list[0] != 0]]
        tab_coeff = np.empty([len(l_s) - 1, len(X.columns)])
        l_s = l_s[1:]

        NB_THREADS = 10

        logging.info("Selecting features according to LASSO logit regression")
        for i in tqdm(range(0, len(l_s), NB_THREADS)):
            threads = [None] * NB_THREADS
            for j in range(NB_THREADS):
                if i + j < len(l_s):
                    threads[j] = threading.Thread(target=lasso_thread, args=(X, y, i + j, tab_coeff, l_s))
                    threads[j].start()

            for j in range(NB_THREADS):
                if i + j < len(l_s): threads[j].join()

        coeff_df = pd.DataFrame(data=tab_coeff[::-1], index=l_s[::-1], columns=X.columns)
        self.selected_cols = coeff_df[coeff_df[coeff_df != 0].count(axis=1) == 15][coeff_df != 0].dropna(axis=1).columns
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        return X[self.selected_cols]