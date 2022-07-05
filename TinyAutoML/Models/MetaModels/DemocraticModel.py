from typing import Union, List

import numpy as np
import pandas as pd

from ..MetaModel import MetaModel

pd.options.mode.chained_assignment = None  # default='warn'


class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.
    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.
    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor

    def __get__(self, obj, owner):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err

            # lambda, but not partial, allows help() to work with update_wrapper
            out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)  # noqa
        else:

            def fn(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            out = lambda *args, **kwargs: fn(*args, **kwargs)  # noqa
        return out


def available_if(check):
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)


class DemocraticModel(MetaModel):
    """
    Hard Voting Classifier

    The Democratic bottleneck estimator makes the trained models vote to decide the output.
    Classes are assumed to be in alphabetical order
    """

    def __init__(
        self,
        comprehensiveSearch: bool = True,
        parameterTuning: bool = True,
        metrics: str = "accuracy",
        nSplits: int = 10,
        voting: str = "soft",
    ):
        super().__init__(comprehensiveSearch, parameterTuning, metrics, nSplits)
        self.voting = voting
        self.nEstimator: int

    def _check_soft_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting}"
            )
        return True

    def _check_hard_voting(self):
        if self.voting == "soft":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting}"
            )
        return True

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> MetaModel:

        super().fit(X, y, **kwargs)

        self.nEstimator = len(self.estimatorPool)  # type: ignore

        return self

    # Overriding sklearn BaseEstimator methods
    def predict(
        self, X: pd.DataFrame, **kwargs
    ) -> Union[pd.Series, np.ndarray, List[np.ndarray]]:
        return (
            np.argmax(self.predict_proportion(X))
            if self.voting == "hard"
            else np.argmax(self.predict_proba(X), axis=1)
        )

    @available_if(_check_hard_voting)
    def predict_proportion(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> list:
        """
        Returns the proportion of model votes for each class
        """
        if self.estimatorPool is None:
            raise ValueError("You must fit the estimator")

        estimatorsPoolOutputs = self.estimatorPool.predict(X, **kwargs)

        prop = float(estimatorsPoolOutputs.iloc[0, :].sum()) / self.nEstimator
        return [1 - prop, prop]

    @available_if(_check_soft_voting)
    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray], **kwargs
    ) -> Union[pd.Series, np.ndarray, List[np.ndarray]]:
        """
        Returns the average model probability per class
        """
        if self.estimatorPool is None:
            raise ValueError("You must fit the estimator")
        estimatorsPoolProbas = self.estimatorPool.predict_proba(X, **kwargs)
        return np.mean(estimatorsPoolProbas, axis=0)

    def transform(self, X: pd.DataFrame):
        return X

    def __repr__(self, **kwargs):
        return "Democratic Model"
