import pandas as pd

from ..Estimator.Estimator import MetaPipeline

from sklearn.datasets import load_breast_cancer


def test_meta_pipeline():
    iris = load_breast_cancer()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = iris.target
    mp = MetaPipeline()
    mp.fit(X, y)
    print(mp.get_scores())
    assert not mp.get_scores().isnull().values.any()
