import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    """
    Load the dataset and preprocess it.
    Args:
    - filepath (str): Path to the CSV file.
    
    Returns:
    - X (DataFrame): Feature matrix.
    - y (Series): Target vector.
    """
    df = pd.read_csv(filepath)
    
    le = LabelEncoder()
    X = df.drop(columns=['reordered'])
    y = df['reordered']
    X['product_name'] = le.fit_transform(X['product_name'])
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
    - X (DataFrame): Feature matrix.
    - y (Series): Target vector.
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)



