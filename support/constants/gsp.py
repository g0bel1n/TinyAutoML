# RANDOM FOREST

n_estimators = [3, 10, 30]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2, 3, 4, 5]

# Method of selecting samples for training each tree
bootstrap = [False]
# Create the random grid
random_forest_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'bootstrap': bootstrap}

xgb_params = {'objective':['binary:logistic'],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'silent' : [True],
        'verbosity' : [0]
        }

estimators_params = {'random forest classifier': random_forest_grid, "xgb": xgb_params}
