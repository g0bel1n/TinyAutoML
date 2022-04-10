import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
print(os.listdir())
from TinyAutoML import MetaPipelineCV
from TinyAutoML.Models import DemocraticModelCV

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

global dm
dm: MetaPipelineCV = MetaPipelineCV(model=DemocraticModelCV())

# For now, the following tests do not test output values but rather if it can run without issues
def test_fit():
    dm.fit(X, y)
    assert True


def test_predict():
    dm.predict(X)
    assert True


def test_transform():
    assert np.all(np.isfinite(dm.transform(X)))


def test_fit_transform():
    assert np.all(np.isfinite(dm.fit_transform(X, y)))



def test_classification_report():
    dm.classification_report(X, y)
    assert True

