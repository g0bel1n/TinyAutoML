import logging
import numpy as np

def extract_score_params(cv_results: dict, n_splits: int, k_last_splits=5) -> tuple[np.float64, dict]:
    """
    :param k_last_splits: k last splits to consider for averaging scores
    :param cv_results: output of GridSearchCV.cv_results_
    :param n_splits: number of splits in the cross_validation
    :return: best score on the last k splits, and the params
    """
    if n_splits < k_last_splits:
        logging.warning(
            'There is more k_last_splits to include than the number of total splits.'
            ' k_last_splits will be set to {0}'.format(
                n_splits))
        k_last_splits = n_splits

    idx_best_config = np.argmin(cv_results['rank_test_score'])
    score = np.mean([cv_results['split{0}_test_score'.format(k)][idx_best_config] for k in
                     range(n_splits, n_splits - k_last_splits)])[0]
    params = cv_results['params'][idx_best_config]
    return score, params