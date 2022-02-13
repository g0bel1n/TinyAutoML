import numpy as np

# RANDOM FOREST

n_estimators = [3,10,30]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,3,4,5]

# Method of selecting samples for training each tree
bootstrap = [False]
# Create the random grid
random_forest_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

# LINEAR DISCRIMINANT ANALYSIS

LDA_grid = {'solver':['svd', 'lsqr']}

estimators_params = {'LDA': LDA_grid, 'rcf' :random_forest_grid}



