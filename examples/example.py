import pandas as pd
import numpy as np
from TinyAutoML.Estimator import MetaPipeline
from TinyAutoML.Models.OneRulerForAll import OneRulerForAll

orfa = OneRulerForAll(False, 'accuracy')
mp = MetaPipeline(model=orfa)

df = pd.read_csv('database.csv')
df.dropna(inplace=True)
y = np.random.choice([0,1],len(df.iloc[:,2]))
df['Date'] = pd.to_datetime(df['Date'])

df.set_index(['Date'], inplace=True)
X = df.iloc[:,2:6]

cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

mp.fit(X_train, y_train)

mp.transform(X,y)
mp.predict(X_test.iloc[:2,:])
mp.classification_report(X_test, y_test)
mp.roc_curve(X_test, y_test)

