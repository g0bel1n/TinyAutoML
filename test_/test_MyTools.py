import pandas as pd
import pytest
from datetime import datetime, timedelta
from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

from support.MyTools import isIndexedByTime, getAdaptedCrossVal, checkClassBalance, buildColumnTransformer, \
    buildMetaPipeline

iris = load_breast_cancer()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target
today = datetime.today()

time_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3],
                        'date': [today, today + timedelta(days=365), today + timedelta(days=700)]}).set_index('date')
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3]})
print(time_df)

target_unbalanced = pd.Series([1, 1, 1, 1, 1, 1, 1, 0])
target_balanced = pd.Series([1, 1, 1, 0, 0, 0])


@pytest.mark.parametrize("input,expected", [(time_df, True), (df, False), ])
def test_is_indexed_by_time(input, expected):
    print(input.index.dtype)
    assert isIndexedByTime(input) == expected


@pytest.mark.parametrize("input,expected", [(time_df, TimeSeriesSplit), (df, StratifiedKFold), ])
def test_get_adapted_cross_val(input, expected):
    assert type(getAdaptedCrossVal(input, 5)) == expected

# For now, the following tests do not test output values but rather if it can run without issues
def test_check_class_balance():
    with pytest.raises(ValueError):
        assert checkClassBalance(target_unbalanced)


def test_build_column_transformer():
    buildColumnTransformer(df)
    assert True


def test_build_meta_pipeline():
    buildMetaPipeline(df, ('dummy', DummyClassifier))
    assert True
