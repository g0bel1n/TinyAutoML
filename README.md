# MetaPipeline
[![Python application](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/g0bel1n/TinyAutoML/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Meta - Pipeline for Stat'App project.
Only Work for binary classification for now.

## Example:

``` python
import pandas as pd

from MetaPipeline import MetaPipeline

from sklearn.datasets import load_breast_cancer

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y=iris.target

mp = MetaPipeline()
mp.fit(X,y)

print(mp.classification_report)

```
