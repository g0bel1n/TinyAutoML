import pandas as pd

from MetaPipeline import estimator

from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y=iris.target


mp = MetaPipeline()
mp.fit(X,y, grid_search=False)

print(mp.classification_report)