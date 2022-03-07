import pandas as pd
from TinyAutoML.Estimator import MetaPipeline
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp = MetaPipeline(model='orfa', gridSearch=False)
mp.fit(X_train, y_train)

print(mp.transform(X_test.iloc[:2,:],y_test[:2]))
mp.predict(X_test.iloc[:2,:])
print(mp.classification_report(X_test, y_test))
mp.roc_curve(X_test, y_test)

