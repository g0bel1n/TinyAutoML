import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from TinyAutoML.Models import *
from TinyAutoML import MetaPipeline

"""iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target
"""


df = pd.read_csv('examples/database.csv')
df.dropna(inplace=True)
y = np.random.choice([0,1],len(df.iloc[:,2]))
df['Date'] = pd.to_datetime(df['Date'])

df.set_index(['Date'], inplace=True)
X = df.iloc[:,2:20]
cut = int(len(y) * 0.8)

X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]


model = BestModel(comprehensiveSearch = False, parameterTuning=False, metrics='accuracy')
mp = MetaPipeline(model=model)
mp.fit(X_train, y_train)
pool = mp.get_pool()

model_2 = DemocraticModel(comprehensiveSearch = False, parameterTuning=False, metrics='accuracy')
mp_2 = MetaPipeline(model=model_2)
mp_2.fit(X_train, y_train, pool=pool)

model_3 = OneRulerForAll(comprehensiveSearch = False, parameterTuning=False, metrics='accuracy')
mp_3 = MetaPipeline(model=model_3)
mp_3.fit(X_train, y_train, pool=pool)

mp_2.classification_report(X_test, y_test)
mp.classification_report(X_test, y_test)
mp_3.classification_report(X_test, y_test)




#mp.transform(X,y)
#mp.predict(X_test)
#mp.classification_report(X_test, y_test)


