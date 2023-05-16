import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LabeledOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom One-Hot Encoder for pandas DataFrame or numpy array.

    This encoder transforms categorical columns into one-hot encoded columns,
    while preserving the continuous columns. It provides labeled column names
    for the new categorical columns in the form of feature_I.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    columns : array-like
        The column names of the original DataFrame or feature names of the array.
    cat_columns : array-like
        The column names or feature names of the categorical columns.
    continuous_columns : array-like
        The column names or feature names of the continuous columns.
    categories_ : dict
        A dictionary containing unique categories for each categorical column.

    Methods:
    --------
    fit(X)
        Fit the encoder to the input DataFrame or array and store necessary attributes.
        Returns the encoder object.
    transform(X)
        Transform the input DataFrame or array by one-hot encoding categorical columns
        and concatenating with continuous columns. Returns the transformed DataFrame.
    """

    def __init__(self):
        self.columns = None
        self.cat_columns = None
        self.continuous_columns = None
        self.categories_ = None

    def fit(self, X, y=None):
        """
        Fit the encoder to the input DataFrame or array.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array, shape (n_samples, n_features)
            The input DataFrame or array to fit the encoder.

        Returns:
        --------
        self : LabeledOneHotEncoder
            The fitted encoder object.
        """
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
            self.cat_columns = X.select_dtypes(include=['object', 'category']).columns
            self.continuous_columns = X.select_dtypes(exclude=['object', 'category']).columns
            self.categories_ = {col: X[col].unique() for col in self.cat_columns}
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1]
            self.columns = [f"feature_{i}" for i in range(n_features)]
            self.cat_columns = np.arange(n_features)
            self.continuous_columns = []
            self.categories_ = {i: np.unique(X[:, i]) for i in range(n_features)}
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array.")

        return self

    def transform(self, X, y=None):
        """
        Transform the input DataFrame or array by one-hot encoding categorical columns
        and concatenating with continuous columns.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array, shape (n_samples, n_features)
            The input DataFrame or array to transform.

        Returns:
        --------
        X_transformed : pandas DataFrame, shape (n_samples, n_features_transformed)
            The transformed DataFrame with one-hot encoded categorical columns
            and continuous columns.
        """
        
        if isinstance(X, pd.DataFrame):
            X_cont = X[self.continuous_columns]

            if self.cat_columns.any():
                X_cat = pd.get_dummies(X[self.cat_columns], prefix=self.cat_columns)
                X_transformed = pd.concat([X_cont, X_cat], axis=1)
            else:
                X_transformed = X_cont
        elif isinstance(X, np.ndarray):
            X_cont = np.delete(X, self.cat_columns, axis=1)

            if self.cat_columns.any():
                X_cat = np.concatenate([pd.get_dummies(X[:, i], prefix=f"feature_{i}") for i in self.cat_columns], axis=1)
                X_transformed = np.concatenate((X_cont, X_cat), axis=1)
                X_transformed = pd.DataFrame(X_transformed, columns=self.columns)
            else:
                X_transformed = pd.DataFrame(X_cont, columns=self.columns)
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array.")
            
        return X_transformed