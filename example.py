import pandas as pd

from examples.example_utils import add_AR_cols, create_binary_box
from TinyAutoML import MetaPipeline
from TinyAutoML.Models import *
from TinyAutoML.Preprocessing.LassoFeatureSelectionParallel import (
    FeatureSelectionParallel,
)

"""iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target
"""
if __name__ == "__main__":

    df = pd.read_csv("examples/database.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        create_binary_box(df, relative_threshold=0.05, box_length=5)
        .set_index("Date")
        .dropna(axis=0)
    )
    df = add_AR_cols(df, 7).dropna(axis=0)

    # X,y = df.drop('Box', axis=1)[:'2020'], df['Box'][:'2020']
    X, y = df.drop("Box", axis=1)[:"2020"], df["Box"][:"2020"].values

    cut = int(len(y) * 0.8)

    X_train, X_test = X.iloc[:cut, :], X.iloc[cut:, :]
    y_train, y_test = y[:cut], y[cut:]

    model = BestModel(
        comprehensiveSearch=False, parameterTuning=False, metrics="accuracy"
    )
    mp = MetaPipeline(model=model)
    mp.fit(X_train, y_train)
    pool = mp.get_pool()

    model_2 = DemocraticModel(
        comprehensiveSearch=False, parameterTuning=False, metrics="accuracy"
    )
    mp_2 = MetaPipeline(model=model_2)
    mp_2.fit(X_train, y_train, pool=pool)

    model_3 = OneRulerForAll(
        comprehensiveSearch=False, parameterTuning=False, metrics="accuracy"
    )
    mp_3 = MetaPipeline(model=model_3)
    mp_3.fit(X_train, y_train, pool=pool)
    print(mp_2.estimator)
    print(mp_3.estimator)
    mp_2.classification_report(X_test, y_test)
    mp.classification_report(X_test, y_test)
    mp_3.classification_report(X_test, y_test)


# mp.transform(X,y)
# mp.predict(X_test)
# mp.classification_report(X_test, y_test)
