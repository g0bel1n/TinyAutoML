# TinyAutoML
[![Tests](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Meta - Pipeline for Stat'App project.
Only works for binary classification for now.

## Example:

``` python
import pandas as pd
from TinyAutoML as MetaPipelineCV
from TinyAutoML.Models import DemocraticModelCV
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

model = DemocraticModelCV(parameterTuning=True, metrics='accuracy')
mp = MetaPipelineCV(model=model)
mp.fit(X_train, y_train)
print(mp.classification_report(X_test, y_test))

```


## Methods available :

``` python

metapipe = tam.Estimator.Metapipeline(model, parameterTuning)

model = 'orfa', 'metamodel' or 'democraticmodel'
parameterTuning: bool

    .predict(self, X: pd.DataFrame)
    
    .transform(self, X: pd.DataFrame, y=None)
    
    .fit_transform(self, X: pd.DataFrame, y: pd.Series)
    
    .get_scores(self)
    
    .classification_report(self, X: pd.DataFrame, y: pd.Series)
    
    .roc_curve(self,X: pd.DataFrame, y:pd.Series)

```

orfa stands for One Ruler For All and is equivalent to the ensemble learning technic called Stacking. The user can pass his own top model. Default is RandomForestClassifier()
metamodel selects the best model amongst a pool of classifiers.

