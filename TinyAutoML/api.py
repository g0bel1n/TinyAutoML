from .MetaPipeline import MetaPipeline
from .Models.MetaModels import BestModel, DemocraticModel, OneRulerForAll

metamodel_names = {
    BestModel: ["best", "bestmodel"],
    DemocraticModel: ["democratic", "democraticmodel", "voting"],
    OneRulerForAll: ["one-ruler-for-all", "onerulerforall", "orfa", "stacking"],
}

reverse_dict_names = {v: k for k, list in metamodel_names.items() for v in list}

def create_pipeline(
    kind: str, 
    comprehensiveSearch: bool = True,
    parameterTuning: bool = True,
    metrics: str = "accuracy",
    nSplits: int = 10
):
    """
    Create a MetaPipeline with a specified kind of MetaModel.

    Parameters
    ----------
    kind : str
        The kind of MetaModel to use. Options are 'best', 'bestmodel', 'democratic', 
        'democraticmodel', 'voting', 'one-ruler-for-all', 'onerulerforall', 'orfa', 'stacking'.
    comprehensiveSearch : bool, optional
        Whether to perform a comprehensive search when training the EstimatorPools. 
        Default is True.
    parameterTuning : bool, optional
        Whether to perform parameter tuning when training the EstimatorPools. 
        Default is True.
    metrics : str, optional
        The metrics to use when evaluating the models. Default is 'accuracy'.
    nSplits : int, optional
        The number of splits for the cross-validation. Default is 10.

    Returns
    -------
    MetaPipeline
        The created MetaPipeline.

    Raises
    ------
    ValueError
        If the specified kind is not recognized.
    """
    model_kind = reverse_dict_names.get(kind.lower(), None)
    
    if model_kind is None:
        raise ValueError(f"Model kind must be one of {list(reverse_dict_names.keys())}")
    
    return MetaPipeline(
        model_kind(
            comprehensiveSearch=comprehensiveSearch,
            parameterTuning=parameterTuning,
            metrics=metrics,
            nSplits=nSplits,
        )
    )