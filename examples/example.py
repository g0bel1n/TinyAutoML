import pandas as pd

from estimator import MetaPipeline
from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp = MetaPipeline()
mp.fit(X_train, y_train, grid_search=False)
print(mp.classification_report(X_test,y_test))


