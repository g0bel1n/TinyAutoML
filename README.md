# TinyAutoML
[![Tests](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/github/license/g0bel1n/TinyAutoML?style=flat-square)
![Python 3.9+](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
![Pypi](https://img.shields.io/pypi/v/TinyAutoML?style=flat-square)
![Size](https://img.shields.io/github/repo-size/g0bel1N/TinyAutoML?style=flat-square)
![commit](https://img.shields.io/github/commit-activity/m/g0bel1n/TinyAutoML?style=flat-square)

Meta - Pipeline for Stat'App project.
Only works for binary classification for now.

## Example:

### See the [introductory notebook](https://github.com/g0bel1n/TinyAutoML/blob/master/introduction-to-Tiny-AutoML.ipynb)

``` python
import pandas as pd
from TinyAutoML as MetaPipelineCV
from TinyAutoML.Models import DemocraticModelCV, OneRulerForAllCV
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

model = DemocraticModelCV(comprehensiveSearch=True, metrics='accuracy')
mp = MetaPipelineCV(model=model)
mp.fit(X_train, y_train)
print(mp.classification_report(X_test, y_test))

# Pool sharing :
trained_pool = mp.get_pool()

model_2 = OneRulerForAllCV(comprehensiveSearch=True, metrics='accuracy')
mp_2 = MetaPipelineCV(model=model_2)
mp_2.fit(X_train, y_train, pool = trained_pool) #Training time way shorter
mp_2.predict(X_test)

```


## Methods available :

``` python

    .predict(X: pd.DataFrame, **kwargs)
    
    .transform(X: pd.DataFrame, y=None)
    
    .fit(X: pd.DataFrame, y: pd.Series, pool = None, **kwargs)
    
    .fit_transform(X: pd.DataFrame, y: pd.Series, **kwargs)
    
    .get_scores()
    
    .get_pool()
    
    .classification_report(X: pd.DataFrame, y: pd.Series)
    
    .roc_curve(X: pd.DataFrame, y:pd.Series)

```

