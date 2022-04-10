import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from TinyAutoML.Estimator import MetaPipeline

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

global mp
mp: MetaPipeline = MetaPipeline(model='orfa', gridSearch=True)

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
    assert type(mp.get_scores()) == str


def test_classification_report():
    mp.classification_report(X, y)
    assert True


def test_roc_curve():
    mp.roc_curve(X, y)
    assert True
