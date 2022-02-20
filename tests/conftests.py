import pandas as pd

from ..TinyAutoML.Estimator import MetaPipeline

from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

def test_metamodel():
    mp = MetaPipeline(model='metamodel', grid_search=False)
    mp.fit(X, y)
    assert not mp.get_scores().isnull().values.any()

def test_orfa():
    mp = MetaPipeline(model='orfa', grid_search=True)
    mp.fit(X, y)
    assert mp.predict(X).any()
