from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)
from .Preprocessing.LassoSelectorTransformer import LassoSelectorTransformer
from .Preprocessing.NonStationarityCorrector import NonStationarityCorrector
from .Preprocessing.LabeledOneHotEncoder import LabeledOneHotEncoder
from .support.MyTools import get_df_scaler


def buildPreprocessingPipeline(scaling_method: str = 'standard') -> Pipeline:
    """
    Create a preprocessing pipeline that:
    - One hot encodes categorical features
    - Stationarizes continuous features
    - Scales features
    - Performs feature selection

    Parameters:
    ----------
    scaling_method : str, default='standard'
        The method to use for scaling. Options are 'standard' or 'minmax'.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        The preprocessing pipeline.
    """
    # Check that the scaling method is valid
    if scaling_method not in ['standard', 'minmax']:
        raise ValueError("scaling_method must be 'standard' or 'minmax'")

    # Use the specified scaling method
    if scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # Create the preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('one_hot_encoder', LabeledOneHotEncoder()),
        ('stationarizer', NonStationarityCorrector()),
        ('scaler', get_df_scaler(scaler)()),
        ('feature_selection', LassoSelectorTransformer(k=15)),
    ])

    return preprocessing_pipeline


def buildMetaPipeline(estimator: BaseEstimator, scaling_method: str = 'standard') -> Pipeline:
    """
    Create a main pipeline that includes the preprocessing pipeline and the estimator.

    Parameters:
    ----------
    scaling_method : str, default='standard'
        The method to use for scaling. Options are 'standard' or 'minmax'.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        The main pipeline.
    """
    # Create the preprocessing pipeline
    preprocessing_pipeline = buildPreprocessingPipeline(scaling_method)

    # Create the main pipeline
    meta_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        (estimator.__repr__(), estimator),
    ])

    return meta_pipeline
