# Importing necessary dependencies
import pandas as pd
import numpy as np
import logging
from config import RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats

logger= logging.getLogger(__name__)

def preprocess_days_since_prior_order(orders):
    try:
        orders['days_since_prior_order'] = orders['days_since_prior_order'].replace(np.nan, -1)
        orders['days_since_prior_order'] = orders['days_since_prior_order'].add(1)
        
        return orders
    
    except Exception as error_1:
        return repr(error_1)


def drop_columns(dataframe: pd.DataFrame, columns_name: str):
    try:
        dataframe = dataframe.drop(columns_name, axis=1)
        logger.info('Column(s) dropped.')
        return dataframe
    
    except Exception as error_2:
        return repr(error_2)


def save_data(dataset, path):
    try: 
        dataset.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")

    except Exception as error_3:
        return repr(error_3)


def merge_datasets(order_products, products, orders, aisles):
    try:
        # Merge order_products with products to get product details in each order
        order_products_details = pd.merge(order_products, products, on='product_id', how='left')
        
        # Merge the result with orders to get user and order details
        dataset = pd.merge(order_products_details, orders, on='order_id', how='left')
        
        # Merge the result with aisles to get aisle details
        dataset = pd.merge(dataset, aisles, on='aisle_id', how='left')
        
        return dataset
    except Exception as error_4:
        return repr(error_4)

def stratified_split(dataset, test_size=0.99, random_state=RANDOM_STATE):
    try:
        # Split the dataset into train and test sets with stratification based on 'user_id'
        train_data, test_data = train_test_split(
            dataset, 
            test_size=test_size, 
            stratify=dataset['user_id'], 
            random_state=random_state
        )

        # Display the size of the training and testing sets
        logger.info(f"Training set size: {train_data.shape[0]} rows")
        logger.info(f"Testing set size: {test_data.shape[0]} rows")
        
        # Display the distribution of user_id in both sets
        logger.info("\nUser ID distribution in training set:")
        logger.info(train_data['user_id'].value_counts(normalize=True))
        
        logger.info("\nUser ID distribution in testing set:")
        logger.info(test_data['user_id'].value_counts(normalize=True))
        
        return train_data, test_data

    except Exception as error_5:
        return repr(error_5)

def target_encode_columns(train_data):
    try:
        train_data['aisle_target_enc'] = train_data['aisle'].map(train_data.groupby('aisle')['reordered'].mean())
        train_data['product_name_target_enc'] = train_data['product_name'].map(train_data.groupby('product_name')['reordered'].mean())
        
        return train_data
    
    except Exception as error_6:
        return repr(error_6)


def bin_days_since_prior_order(train_data):
    try:
        bins = [-1, 6, 12, 18, 24, float('inf')]
        labels = ['undefined', 'recently_ordered', 'moderately_ordered', 'infrequently_ordered', 'least_ordered']
        train_data['encoded_days_since_prior_order'] = pd.cut(train_data['days_since_prior_order'], bins=bins, labels=labels)
        
        return train_data
    
    except Exception as error_7:
        return repr(error_7)
