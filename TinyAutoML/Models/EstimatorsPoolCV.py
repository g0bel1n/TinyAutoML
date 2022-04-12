import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union
from numpy import ndarray
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     TimeSeriesSplit)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from  ..constants.gsp import estimators_params
from  ..builders import buildMetaPipeline

pd.options.mode.chained_assignment = None  # default='warn'

class EstimatorPoolCV():

    def __init__(self):

        self.estimatorsList = [("random forest classifier", RandomForestClassifier()),
                               ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                               ('Gaussian Naive Bayes', GaussianNB()),
                               ('LDA', LinearDiscriminantAnalysis()),
                               ('xgb', XGBClassifier(use_label_encoder=False))
                               ]

        self.is_fitted = False
        self.estimatorsPipeline: list[tuple[str,Pipeline]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, Pipeline]]:
        self.estimatorsPipeline = [(estimator_name, buildMetaPipeline(X, estimator).fit(X,y)) for estimator_name, estimator in self.estimatorsList]
        self.is_fitted = True
        return self.estimatorsPipeline

    def fitWithparameterTuning(self, X: pd.DataFrame, y: pd.Series,
                          cv: Union[TimeSeriesSplit, StratifiedKFold],
                          metrics) -> list[tuple[str, Pipeline]]:

        for estimator in tqdm(self.estimatorsList):
            pipe = buildMetaPipeline(X,estimator=estimator[1])
            if estimator[0] in estimators_params:
                grid = {f'{estimator[1].__repr__()}__{key}': value for key, value in estimators_params[estimator[0]].items()}
                clf = RandomizedSearchCV(estimator=pipe,
                                         param_distributions=grid, scoring=metrics,
                                         n_jobs=-2, cv=cv, verbose=1, n_iter=min(10, len(grid)))
                clf.fit(X, y)

                pipe.set_params(**clf.best_params_)

            pipe.fit(X, y)
            self.estimatorsPipeline.append((estimator[0], pipe))
            
        self.is_fitted = True
        return self.estimatorsPipeline

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {estimator_name[0]: pipe.predict(X) for estimator_name, pipe in self.estimatorsPipeline})
        
    def predict_proba(self, X: pd.DataFrame) -> ndarray:
        return np.array([pipe.predict_proba(X) for _,pipe in self.estimatorsPipeline])

    def get_best(self, X: pd.DataFrame, y: pd.Series) -> tuple[float, str, Pipeline]:

        scores = [accuracy_score(pipe.predict(X), y) for _,pipe in self.estimatorsPipeline]
        return float(np.max(scores)), *self.estimatorsPipeline[np.argmax(scores)]

    def get_scores(self, X: pd.DataFrame, y: pd.Series) -> list[tuple[str, float]]: 
        return [(estimator_name, accuracy_score(pipe.predict(X), y)) for estimator_name,pipe in self.estimatorsPipeline]


    def __len__(self):
        return len(self.estimatorsPipeline)
