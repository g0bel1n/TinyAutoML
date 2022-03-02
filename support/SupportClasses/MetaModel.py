import logging
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ..MyTools import extract_score_params
from ..constants.GLOBAL_PARAMS import WINDOW
from ..constants.gsp import estimators_params


class MetaModel(BaseEstimator):

    def __init__(self, grid_search: bool, metrics: str, n_splits=10):
        self.best_estimator_ = None
        self.best_estimator_index = None

        # Pool of estimators
        self.estimators = [("random forest classifier", RandomForestClassifier()),
                           ("Logistic Regression", LogisticRegression(fit_intercept=True)),
                           ('Gaussian Naive Bayes', GaussianNB()),
                           ('LDA', LinearDiscriminantAnalysis()),
                           ('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                           n_estimators=200,
                                                           algorithm="SAMME.R",
                                                           learning_rate=0.5)),
                           ('xgb', xgb.XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                                                     silent=True, use_label_encoder=False, verbosity=0))]
        self.scores = pd.DataFrame()
        self.n_splits = n_splits
        self.grid_search = grid_search
        self.metrics = metrics


    def fit(self, X: pd.DataFrame, y: pd.Series, ) -> BaseEstimator:

        logging.info("Training models")

        # Si on peut découper les WINDOW - premiers features dans la pipeline, qui sont des NaN suite
        # aux fenêtres glissantes de StationarityCorrector, on ne peut pas le faire pour la target.
        # On slice donc ici
        y_train = y[WINDOW:]

        # Pour détécter une distribution déséquilibrée...
        assert len(y[y == 1]) / len(y) <= 0.7, "The target is unbalanced"

        if X.index.dtype != datetime:
            # Si les données ne sont pas temporelles, on applique un découpage stratifié en prenant des
            cv = StratifiedKFold(n_splits=self.n_splits)
        else:
            # Pour des données temporelles, on préféra découper linéairement dans le temps
            cv = TimeSeriesSplit(n_splits=self.n_splits)

        results = {}
        if self.grid_search:
            for estimator in self.estimators:
                logging.info("\t----> {0}".format(estimator[0]))
                # Tous les estimateurs ne sont pas forcément à fine-tuner, on vérifie donc
                # que l'on ait les grilles disponibles.
                logging.basicConfig(level=logging.ERROR)

                if estimator[0] in estimators_params:
                    clf = RandomizedSearchCV(estimator=estimator[1],
                                             param_distributions=estimators_params[estimator[0]], scoring=self.metrics,
                                             n_jobs=-1, cv=cv, refit=True)
                    clf.fit(X, y_train)
                    results[estimator[0]] = extract_score_params(clf.cv_results_, n_splits=self.n_splits,
                                                                 k_last_splits=5)
                else:
                    cvs = cross_val_score(estimator[1], X, y_train, cv=cv, scoring=self.metrics, n_jobs=-1)
                    results[estimator[0]] = (cvs.mean(), None)
                logging.basicConfig(level=logging.INFO)


            self.scores = pd.DataFrame(list(results.values())[:][0],
                                       index=[estimator[0] for estimator in self.estimators])

        else:
            self.scores = pd.DataFrame(
                data=[cross_val_score(estimator[1], X, y_train, cv=cv, scoring=self.metrics, n_jobs=-1) for estimator in
                      self.estimators], index=[estimator[0] for estimator in self.estimators],
                columns=['Split {0}'.format(i) for i in range(self.n_splits)])
            self.scores['mean'] = self.scores.mean(axis=1)

        # Getting the best estimator according to the mean accuracy
        self.best_estimator_index = self.scores['mean'].argmax()
        self.best_estimator_ = self.estimators[self.best_estimator_index][1]

        logging.info("The best estimator is {0} with a cross-validation accuracy (in Sample) of {1}".format(
            self.estimators[self.best_estimator_index][0], self.scores['mean'].iloc[self.best_estimator_index]))

        try:
            params = results[self.estimators[self.best_estimator_index][0]][1]
            self.best_estimator_.set_params(**params)
        except (NameError, KeyError): # if the chosen estimator was not grid searched, the results dict does not exist
            pass

        self.best_estimator_.fit(X, y_train)

        return self

    # Overloading sklearn BaseEstimator methods to use the best estimator
    def predict(self, X: pd.Series) -> pd.Series:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: pd.Series) -> pd.Series:
        return self.best_estimator_.predict_proba(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
