# MetaPipeline
[![Python application](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Meta - Pipeline for Stat'App project.
Only Work for binary classification for now.

## Example:

``` python
import pandas as pd
import TinyAutoML as tam
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp = tam.Estimator.MetaPipeline()
mp.fit(X_train, y_train, grid_search=False)
print(mp.classification_report(X_test, y_test))

```
