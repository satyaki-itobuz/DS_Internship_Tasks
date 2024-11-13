
import pandas as pd

def load_data(file_path:str) -> pd.DataFrame:
    """
    Loading dataset from a CSV file
    """
    try:
        data = pd.read_csv(filepath_or_buffer = file_path)
        return data
    
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return

def clean_data(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Performing basic data cleaning by dropping the specified column
    """
    try:
        cleaned_df = df.drop(columns = [col]) 
        return cleaned_df
    
    except Exception as DataCleaningError:
        return repr(DataCleaningError)

def merging_data(df1:pd.DataFrame, df2:pd.DataFrame, on_column=''):
    """
    Merging two datasets on the specified column
    """
    if (on_column == ''):
        return "Common column not found"
    
    else:
        try:
            df = pd.merge(df1, df2, how='inner', on=on_column)  
            return df
        
        except KeyError:
            print(f"Error: Column '{on_column}' not found in both dataframes.")
            return None

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
