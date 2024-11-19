
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