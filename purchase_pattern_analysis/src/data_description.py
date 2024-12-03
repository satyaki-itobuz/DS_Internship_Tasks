# #### Importing Libraries and Datasets
import pandas as pd

def data_head(dataframe: pd.DataFrame):
    """Display the first 5 rows of the dataset"""
    return dataframe.head()

def data_unique(dataframe: pd.DataFrame):
    """Calculating total unique values for each column"""
    return dataframe.nunique()

def data_null(dataframe: pd.DataFrame):
    """Calculating the total missing values for each column"""
    return dataframe.isna().sum()

def data_duplicate(dataframe: pd.DataFrame):
    """Calculating if there is any duplicate values in the dataset"""
    return dataframe.duplicated().sum()