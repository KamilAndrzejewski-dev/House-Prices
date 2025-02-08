from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List, Dict, Union


class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Transforms datetime variables by subtracting a reference datetime variable.

    Attributes:
        variables_ (List[str]): List of variables.
        reference_str_ (str): Name of variable to be subtracted from datetime variables.

    """

    def __init__(self, variables: List[str], reference_str: str) -> None:
        """Initializes the TemporalVariableTransformer

        Args:
            variables (List[str]): List of datetime variables.
            reference_str (str): Reference datetime variable to be subtracted.

        Raises:
            TypeError: If variables is not a list or reference_str a string.
        """

        if not isinstance(variables, list):
            raise TypeError("Variables should be a list")
        if not isinstance(reference_str, str):
            raise TypeError("Reference should be a string")

        self.variables_ = variables
        self.reference_str_ = reference_str

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TemporalVariableTransformer':
        """This step doesn't perform any action.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Input Series.

        Returns:
            TemporalVariableTransformer: Returns fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Subtract reference variable with datetime column.

        Args:
            X (pd.dataFrame): Input DataFrame

        Returns:
            DataFrame: Returns DataFrame with transformed datetime variables.

        Raises:
            KeyError: If column wasn't found in DataFrame
        """
        X = X.copy()

        for col in self.variables_:
            if isinstance(col, str) and col in X.columns:
                X[col] = X[self.reference_str_] - X[col]
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return X


class CustomSimpleImpute(BaseEstimator, TransformerMixin):
    """
    Impute missing values using specified method.

    Attributes:

        variables_ (List[str]): List of variables.

        imputation_ (str): Imputation method (median, mean, constant, most_frequent).

        fill_values_ (str): The fixed value used for filling missing value.

        encoder_ (Dict[str, Union[str, float, int]]): Dictionary storing variable names and values corresponding to imputed values used for imputation.
    """

    def __init__(self, variables: List[str], imputation: str = "mean", fill_values: Union[int, float, str] = "Missing"):
        """ Initializes the CustomSimpleImpute

        Args:
            variables (List[str]): List of variables to be imputed.
            imputation (str): Imputation method.
            fill_values (Union[int, float, str]): The fixed value used for filling missing values when imputation method is "constant".

        Raises:
            TypeError: If variables aren't a list, imputation a string or fill_value a (int, float, str)
            ValueError: If imputation isn't a (int, float, str)
        """
        if not isinstance(variables, list):
            raise TypeError("Variables must be a list")
        if not isinstance(imputation, str):
            raise TypeError("imputation must be a str")
        if imputation not in ["mean", "median", "constant", "most_frequent"]:
            raise ValueError("imputation must be following values ['mean', 'median', 'constant', 'most_frequent']")
        if not isinstance(fill_values, (int, float, str)):
            raise TypeError("fill_value must best int, float or str")

        self.variables_ = variables
        self.imputation_ = imputation
        self.fill_values_ = fill_values
        self.encoder_: Dict[str, Union[str, float, int]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomSimpleImpute':
        """ Calculates the central tendency values for each variable.
        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Input Series.

        Returns:
            CustomSimpleImpute: Returns fitted transformer.

        Raises:
            KeyError: If column wasn't found in Dataframe
            TypeError: If column of a specific type can't be used with specific imputation
        """
        for col in self.variables_:
            if col not in X.columns:
                raise KeyError(f"Column {col} not find in DataFrame")

            if X[col].dtypes in ['float64', 'int64']:
                if self.imputation_ == "mean":
                    self.encoder_[col] = round(X[col].mean(), 1)
                elif self.imputation_ == "median":
                    self.encoder_[col] = round(X[col].median(), 1)
                else:
                    raise TypeError(f"Column {col} of type {X[col].dtypes} cannot be used with {self.imputation_}")

            elif X[col].dtypes in ['object', 'category']:
                if self.imputation_ == "most_frequent":
                    self.encoder_[col] = X[col].mode()[0]
                elif self.imputation_ == "constant":
                    self.encoder_[col] = self.fill_values_
                else:
                    raise TypeError(f"Column {col} of type {X[col].dtypes} cannot be used with {self.imputation_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values with their corresponding imputation values stored in encoder.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            DataFrame: Returns DataFrame with missing values filler for specified columns.

        Raises:
            KeyError: If column specified in variables wasn't found in DataFrame.
        """
        for col in self.variables_:
            if col in X.columns:
                X[col] = X[col].fillna(self.encoder_[col])
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Maps categorical variables to numeric values.

    Attributes:
        variables_ (List[str]): List of categorical variables
        mapping_ (Dict[str, int]): Dictionary with mapping.
    """

    def __init__(self, variables: List[str], mapping: Dict[str, int]) -> None:
        """ Initializes Mapper

        Args:
            variables (List[str]): List of categorical variables.
            mapping (Dict[str, int]): Dictionary with specified mapping.

        Raises:
            TypeError: If variables is not a list or mapping a dictionary.
            ValueError: If mapping keys aren't string or values aren't integers
        """
        self.variables_ = variables
        self.mapping_ = mapping

        if not isinstance(variables, list):
            raise TypeError("variables is not a list")
        if not isinstance(mapping, dict):
            raise TypeError("mapping is not a dictionary")
        if not all(isinstance(keys, str) for keys in mapping.keys()):
            raise ValueError("mapping key must be a string")
        if not all(isinstance(values, int) for values in mapping.values()):
            raise ValueError("mapping value must be an integer")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'Mapper':
        """This step doesn't perform any action.
        Args:
            X (pd.DataFrame): Input DataFrame
            y (pd.Series): Input Series

        Returns:
            Mapper: Returns fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Maps variables to their corresponding numeric values.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            DataFrame: Returns DataFrame with mapped variables values.

        Raises:
            Keyword: If variable specified in variables_ wasn't found in DataFrame.
        """
        X = X.copy()
        for col in self.variables_:
            if col in X.columns:
                X[col] = X[col].map(self.mapping_).fillna(0).astype(int)
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return X


class RareLabelsEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categories with frequency below a threshold as Rare.

    Attributes:
        variables_ (List[str]): List of a variables
        threshold_ (float): The minimum frequency required for a variable to be included in a returned list. By default, is equal to 0.01.
        encoder_ (dict): Dictionary with variables names and corresponding values divided into saved and changed to 'Rare'.
    """

    def __init__(self, variables: List[str], threshold: float = 0.01) -> None:
        """ Initializes RareLabelsEncoder

        Args:
            variables (List[str]): List of categorical variables.
            threshold (float): The minimum frequency threshold for value to be included. Below this threshold, value is change to 'Rare'.
        Raises:
            TypeError: If variables is not a list or threshold a float.
            ValueError: If a threshold is lower than 0 or greater than 1.
        """
        self.variables_ = variables
        self.threshold_ = threshold
        self.encoder_: dict = {}

        if not isinstance(variables, list):
            raise TypeError("Variables is not a list")
        if not isinstance(threshold, float):
            raise TypeError("Threshold must be a floating number")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Threshold must be between 0 and 1")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RareLabelsEncoder':
        """Calculates frequency of each value in specified variable
           and save values that are above a threshold

        Args:
            X (pd.DataFrame): Input Pandas.
            y (pd.Series): Input Series.

        Returns:
            RareLabelsEncoder: Returns fitted transformer.

        Raises:
            KeyError: If column specified in variables_ is not in DataFrame.
        """
        X_copy = X.copy()

        for col in self.variables_:
            if col in X.columns:
                rare = (
                    pd.Series(
                        X_copy[col].value_counts(normalize=True)
                    )
                )
                self.encoder_[col] = (
                    list(rare[rare >= self.threshold_].index)
                )
            else:
                raise KeyError(f"Column {col} is missing")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Maps variables that aren't in encoder as 'Rare'

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            DataFrame: Returns DataFrame with mapped variables.

        Raises:
            KeyError: If column specified in variables_ wasn't found in DataFrame.
        """
        X = X.copy()
        for col in self.variables_:
            if col in X.columns:
                X[col] = (
                    np.where(
                        X[col].isin(
                            self.encoder_[col]), X[col], 'Rare'
                    )
                )
            else:
                raise KeyError(f"Column {col} is missing")
        return X


class MonotonicOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categories based on target in ascending order.

    Attributes:
        variables_ (List[str]): List of variables.
        method_ (str): Measures of a central tendency.
        encoder_ (Dict[str, dict]): Dictionary storing variables names as keys and numbers as values in monotonic order base on Target.
    """

    def __init__(self, variables: List[str], method: str = 'mean') -> None:
        """Initializes MonotonicOrdinalEncoder

        Args:
            variables (List[str]): List of variables.
            method (str): Measures of a central tendency.

        Raises:
            TypeError: If variables is not a list or method a string.
            ValueError: If method is not equal to 'mean' or 'median'.
        """
        self.variables_ = variables
        self.method_ = method
        self.encoder_: Dict[str, dict] = {}
        self.target = None

        if not isinstance(variables, list):
            raise TypeError('variables must be a list')
        if not isinstance(method, str):
            raise TypeError('method must be of type string')
        if self.method_ not in ['mean', 'median']:
            raise ValueError('method must be "median" or "mean"')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MonotonicOrdinalEncoder':
        """ Calculates the mean or median of each category and sorts the categories
        in ascending order, then it assigns integer encodings to the categories.
        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Input Series.

        Returns:
            MonotonicOrdinalEncoder: Returns fitted transformer.
        """


        dataframe = pd.concat([X, y], axis=1)
        sorted_variables = None
        for col in self.variables_:
            if col in dataframe.columns:
                if self.method_ == 'mean' and col != 'SalePrice':
                    sorted_variables = (
                        dataframe.groupby([col])[y.name].mean().sort_values(ascending=True)
                    )
                elif self.method_ == 'median' and col != 'SalePrice':
                    sorted_variables = (
                        dataframe.groupby([col])[y.name].median().sort_values(ascending=True)
                    )

                encoding = {k: i for i, k in enumerate(sorted_variables.index)}
                self.encoder_[col] = encoding
                self.target = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Maps integer encodings to the categories in ascending order.

        Args:
            X (pd.DataFrame): Pandas DataFrame.

        Returns:
            DataFrame: Returns DataFrame with mapped variables.

        Raises:
            KeyError: If columns specified in variables weren't found in DataFrame.
        """
        dataframe = X.copy()
        for col in self.variables_:
            if col in dataframe.columns and col != self.target.name:
                dataframe[col] = dataframe[col].map(self.encoder_[col])

        return dataframe


class MathFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a mathematical function to specified variables.

    Attributes:
        variables_ (List[str]): List of variables.
        func_ (str): Name of the function to apply ('log', 'sqrt', 'exp').
    """

    def __init__(self, variables: List[str], func: str) -> None:
        """ Initializes MathFunctionTransformer

        Args:
            variables (List[str]): List of variables.
            func: Function to apply transformation ('log', 'sqrt', 'exp').

        Raises:
            TypeError: If variables is not a list or func a string.
            ValueError: If func is not equal to 'log', 'exp' or 'sqrt'.
        """
        if not isinstance(variables, list):
            raise TypeError("Variables must be a list")
        if not isinstance(func, str):
            raise TypeError("Func must be a str")
        if func not in ['log', 'exp', 'sqrt']:
            raise ValueError("Function not supported. Choose from ['log', 'exp', 'sqrt']")

        self.variables_ = variables
        self.func_ = func

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'MathFunctionTransformer':
        """ This step doesn't perform any action

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Input Series.
        Returns:
            MathFunctionTransformer: Returns fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Transform specified variables values base on specified function

        Args:
            X (pd.DataFrame): Input Pandas.
        Returns:
            DataFrame: Returns DataFrame with transformed variables values.
        Raises:
            KeyError: If column specified in variables_ wasn't found in DataFrame.
        """
        X = X.copy()
        for col in self.variables_:
            if col in X.columns:
                if self.func_ == 'log':
                    X[col] = np.log(X[col])
                elif self.func_ == 'exp':
                    X[col] = np.exp(X[col])
                elif self.func_ == 'sqrt':
                    X[col] = np.sqrt(X[col])
            else:
                raise KeyError(f"Column {col} does not exist")
        return X


class CustomBinarizer(BaseEstimator, TransformerMixin):
    """
    Binarizes variables values based on a specified threshold.

    Attributes:
        variables_ (List[str]): List of variables
        threshold_ (Union[int, float]): Threshold value for binarization.
    """

    def __init__(self, variables: List[str], threshold: Union[int, float] = 0):
        """ Initializes CustomBinarizer

        Args:
            variables (List[str]): List of variables.
            threshold (Union[int, float]): Threshold value for binarizartion. By default, its equal to 0.

        Raises:
             TypeError: If variables is not a list or threshold a number.
             ValueError: If a threshold is lower than 0 or greater than 1.
        """
        if not isinstance(variables, list):
            raise TypeError("variables must be a list")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be an int or float")
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be greater than 0 or lower than 1")

        self.variables_ = variables
        self.threshold_ = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomBinarizer':
        """This step doesn't perform any action

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Input Series.

        Returns:
            CustomBinarizer: Returns fitted transformer.
        """
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Binarize values to 1. Values equal to 0 remain unchanged.

        Args:
            X (pd.DataFrame): Input DataFrame

        Returns:
            DataFrame: Returns DataFrame with binarized variables.
        """
        X = X.copy()
        for col in self.variables_:
            if col in X.columns:
                X[col] = np.where(X[col] > 0, 1, 0)
            else:
                raise KeyError(f"Column {col} does not exist")
        return X
