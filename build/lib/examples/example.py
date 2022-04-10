import pandas as pd
import numpy as np
from TinyAutoML.Estimator import MetaPipeline
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
X = pd.DataFrame(data=ds.data, columns=ds.feature_names)
y = ds.target
df = pd.read_csv('database.csv')
df.dropna(inplace=True)
df['y'] = np.random.choice([0,1],len(df.iloc[:,2]))
df['Date'] = pd.to_datetime(df['Date'])

df.set_index(['Date'], inplace=True)
X = df.iloc[:,2:6]

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp = MetaPipeline(model='orfa', gridSearch=False)
mp.fit(X, df.y)

print(mp.transform(X,df.y))
mp.predict(X_test.iloc[:2,:])
print(mp.classification_report(X_test, y_test))
mp.roc_curve(X_test, y_test)

