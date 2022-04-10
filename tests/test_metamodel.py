import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from TinyAutoML import MetaPipeline
from TinyAutoML.Models import BestModel

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

global mp
mp: MetaPipeline = MetaPipeline(model=BestModel())


# For now, the following tests do not test output values but rather if it can run without issues
def test_fit():
    mp.fit(X, y)
    assert True


def test_predict():
    mp.predict(X)
    assert True


def test_transform():
    assert np.all(np.isfinite(mp.transform(X)))


def test_fit_transform():
    assert np.all(np.isfinite(mp.fit_transform(X, y)))


def test_get_scores():
    assert np.all(np.array(mp.get_scores(X,y))!=0)


def test_classification_report():
    mp.classification_report(X, y)
    assert True

