# #### Importing Libraries and Datasets
import pandas as pd

def data_head(dataframe: pd.DataFrame):
    try:
        """Display the first 5 rows of the dataset"""
        return dataframe.head()
    except Exception as unable_to_check_head:
        return repr(unable_to_check_head)

def data_unique(dataframe: pd.DataFrame):
    try:
        """Calculating total unique values for each column"""
        return dataframe.nunique()
    except Exception as unable_to_check_nunique:
        return repr(unable_to_check_nunique)

def data_null(dataframe: pd.DataFrame):
    try:
        """Calculating the total missing values for each column"""
        return dataframe.isna().sum()
    except Exception as unable_to_check_null:
        return repr(unable_to_check_null)

def data_duplicate(dataframe: pd.DataFrame):
    try:
        """Calculating if there is any duplicate values in the dataset"""
        return dataframe.duplicated().sum()
    except Exception as unable_to_check_duplicate:
        return repr(unable_to_check_duplicate)