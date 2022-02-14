import pandas as pd
from Estimator.Estimator import MetaPipeline
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp = MetaPipeline()
mp.fit(X_train, y_train, grid_search=False)
print(mp.classification_report(X_test, y_test))


