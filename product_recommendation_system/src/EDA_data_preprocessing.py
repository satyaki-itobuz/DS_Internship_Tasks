
import pandas as pd
import logging

# Configure logger 
logger = logging.getLogger(__name__)


def clean_data(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Performing basic data cleaning by dropping the specified column
    """
    try:
        cleaned_df = df.drop(columns = [col]) 
        logger.info("Basic data cleaning")
        return cleaned_df
    
    except Exception as DataCleaningError:
        logger.error("An error occured")
        return repr(DataCleaningError)

def fill_missing(df:pd.DataFrame, col:str, fill_value:int) -> pd.DataFrame:
    """
    Fill missing values for a specific column with a fill_value
    """

    try:
        df[col] = df[col].fillna(fill_value)
        return df
    
    except Exception as ColumnNotFound:
        logger.error("Column not found")
        return repr(ColumnNotFound)

def missing_values(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Handling missing values by filling with -1 and applying a transformation
    """
    try:
        df = fill_missing(df, col, -1)
        df[col] = df[col].apply(lambda x: x + 1)
        return df
    
    except Exception as Wrongvalue:
        logger.error("Column not found")
        return repr(Wrongvalue)
