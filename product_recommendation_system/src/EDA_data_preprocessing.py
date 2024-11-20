
import pandas as pd


def clean_data(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Performing basic data cleaning by dropping the specified column
    """
    try:
        cleaned_df = df.drop(columns = [col]) 
        return cleaned_df
    
    except Exception as DataCleaningError:
        return repr(DataCleaningError)

def fill_missing(df:pd.DataFrame, col:str, fill_value:int) -> pd.DataFrame:
    """
    Fill missing values for a specific column with a fill_value
    """
    df[col] = df[col].fillna(fill_value)
    return df

def missing_values(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Handling missing values by filling with -1 and applying a transformation
    """
    df = fill_missing(df, col, -1)
    df[col] = df[col].apply(lambda x: x + 1)
    return df
