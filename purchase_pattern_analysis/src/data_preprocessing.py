# Importing necessary dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats

def preprocess_days_since_prior_order(orders):
    orders['days_since_prior_order'] = orders['days_since_prior_order'].replace(np.nan, -1)
    orders['days_since_prior_order'] = orders['days_since_prior_order'].add(1)
    
    return orders


def drop_columns(dataframe: pd.DataFrame, columns_name: str):
    dataframe = dataframe.drop(columns_name, axis=1)
    return dataframe


def save_data(dataset, path):
    dataset.to_csv(path, index=False)
    print(f"Data saved to {path}")


def merge_datasets(order_products, products, orders, aisles):
    # Merge order_products with products to get product details in each order
    order_products_details = pd.merge(order_products, products, on='product_id', how='left')
    
    # Merge the result with orders to get user and order details
    dataset = pd.merge(order_products_details, orders, on='order_id', how='left')
    
    # Merge the result with aisles to get aisle details
    dataset = pd.merge(dataset, aisles, on='aisle_id', how='left')
    
    return dataset


def stratified_split(dataset, test_size=0.99, random_state=42):
    # Split the dataset into train and test sets with stratification based on 'user_id'
    train_data, test_data = train_test_split(
        dataset, 
        test_size=test_size, 
        stratify=dataset['user_id'], 
        random_state=random_state
    )
    
    # Display the size of the training and testing sets
    print(f"Training set size: {train_data.shape[0]} rows")
    print(f"Testing set size: {test_data.shape[0]} rows")
    
    # Display the distribution of user_id in both sets
    print("\nUser ID distribution in training set:")
    print(train_data['user_id'].value_counts(normalize=True))
    
    print("\nUser ID distribution in testing set:")
    print(test_data['user_id'].value_counts(normalize=True))
    
    return train_data, test_data


def target_encode_columns(train_data):
    train_data['aisle_target_enc'] = train_data['aisle'].map(train_data.groupby('aisle')['reordered'].mean())
    train_data['product_name_target_enc'] = train_data['product_name'].map(train_data.groupby('product_name')['reordered'].mean())
    
    return train_data


def bin_days_since_prior_order(train_data):
    bins = [-1, 6, 12, 18, 24, float('inf')]
    labels = ['undefined', 'recently_ordered', 'moderately_ordered', 'infrequently_ordered', 'least_ordered']
    train_data['encoded_days_since_prior_order'] = pd.cut(train_data['days_since_prior_order'], bins=bins, labels=labels)
    
    return train_data
