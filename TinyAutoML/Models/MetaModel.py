import logging
from typing import Union

import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from TinyAutoML.Models.EstimatorsPool import EstimatorPool
from TinyAutoML.support.MyTools import extract_score_params, getAdaptedCrossVal, checkClassBalance
from TinyAutoML.constants.gsp import estimators_params


class MetaModel(BaseEstimator):

    def __init__(self, grid_search=True, metrics='accuracy', n_splits=10):
        self.best_estimator_name = None
        self.best_estimator_ = None
        self.best_estimator_index = None

        # Pool of estimators
        self.estimatorPool = EstimatorPool()
        self.scores = pd.DataFrame()
        self.n_splits = n_splits
        self.grid_search = grid_search
        self.metrics = metrics

    def __fitPool(self, X: pd.DataFrame, y: pd.Series, cv: Union[TimeSeriesSplit, StratifiedKFold]) -> list[
        tuple[str, BaseEstimator]]:

        self.scores = pd.DataFrame(
            data=[cross_val_score(estimator[1], X, y, cv=cv, scoring=self.metrics, n_jobs=-1) for estimator in
                  self.estimatorsList], index=[estimator[0] for estimator in self.estimatorsList],
            columns=['Split {0}'.format(i) for i in range(self.n_splits)])
        self.scores['mean'] = self.scores.mean(axis=1)

        return self.estimatorsList

    def __fitPoolGridSeach(self, X: pd.DataFrame, y: pd.Series, cv: Union[TimeSeriesSplit, StratifiedKFold]) -> tuple[
        dict, list[tuple[str, BaseEstimator]]]:
        results = {}

        for estimator in self.estimatorsList:
            logging.info("\t----> {0}".format(estimator[0]))
            # Tous les estimateurs ne sont pas forcément à fine-tuner, on vérifie donc
            # qu'on ait les grilles disponibles.
            logging.basicConfig(level=logging.ERROR)

            if estimator[0] in estimators_params:
                clf = RandomizedSearchCV(estimator=estimator[1],
                                         param_distributions=estimators_params[estimator[0]], scoring=self.metrics,
                                         n_jobs=-1, cv=cv, refit=True)
                clf.fit(X, y)
                results[estimator[0]] = extract_score_params(clf.cv_results_, n_splits=self.n_splits,
                                                             k_last_splits=5)
            else:
                cvs = cross_val_score(estimator[1], X, y, cv=cv, scoring=self.metrics, n_jobs=-1)
                results[estimator[0]] = (cvs.mean(), None)
            logging.basicConfig(level=logging.INFO)

        self.scores = pd.DataFrame(list(results.values())[:][0],
                                   index=[estimator[0] for estimator in self.estimatorsList])

        return results, self.estimatorsList

    def fit(self, X: pd.DataFrame, y: pd.Series, ) -> BaseEstimator:


        logging.info("Training models")

        # Pour détecter une distribution déséquilibrée...
        checkClassBalance(y)
        # Récupération d'un split de CV adapté selon l'indexage du set
        cv = getAdaptedCrossVal(X, self.n_splits)

        if self.grid_search:
            self.estimatorPool.fitWithGridSearch(X,y,cv,'accuracy')
        else:
            self.estimatorPool.fit(X, y)

        # Getting the best estimator according to the metric mean
        best_score , self.best_estimator_name, self.best_estimator = self.estimatorPool.get_best(X,y)

        logging.info("The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
            self.best_estimator_name, best_score))

        return self

    # Overloading sklearn BaseEstimator methods to use the best estimator
    def predict(self, X: pd.Series) -> pd.Series:
        return self.best_estimator.predict(X)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        return self.best_estimator.predict_proba(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def __repr__(self, **kwargs):
        return 'Meta Model'
