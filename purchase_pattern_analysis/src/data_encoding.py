import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def apply_target_encoding(data, column, target_column):
    """
    Apply target encoding on the specified column.
    """
    return data[column].map(data.groupby(column)[target_column].mean())

def csv_save(df, output_folder, filename):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - output_folder (str): Path to the output folder.
    - filename (str): Name of the CSV file.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def preprocess_data(input_file, output_file):
    """
    Preprocess the sampled data:
    - Target encoding
    - One-hot encoding
    - Cyclic encoding
    - Ordinal encoding
    - Binning
    """
    try:
        # Load the sampled data
        sampled_data = pd.read_csv(input_file)
        
        # Apply Target Encoding
        sampled_data['aisle_target_enc'] = apply_target_encoding(sampled_data, 'aisle', 'reordered')
        sampled_data['department_target_enc'] = apply_target_encoding(sampled_data, 'department', 'reordered')
        sampled_data['product_name_target_enc'] = apply_target_encoding(sampled_data, 'product_name', 'reordered')

        # Apply One-hot Encoding on 'order_dow'
        day_of_week_dummies = pd.get_dummies(sampled_data['order_dow'], prefix='dow').astype(int)
        sampled_data = pd.concat([sampled_data, day_of_week_dummies], axis=1)

        # Apply Cyclic Encoding on 'order_hour_of_day'
        sampled_data['order_hour_sin'] = np.sin(2 * np.pi * sampled_data['order_hour_of_day'] / 24)
        sampled_data['order_hour_cos'] = np.cos(2 * np.pi * sampled_data['order_hour_of_day'] / 24)

        # Apply Ordinal Encoding on 'add_to_cart_order'
        ordinal_encoder = OrdinalEncoder()
        sampled_data['add_to_cart_order_encoded'] = ordinal_encoder.fit_transform(
            sampled_data[['add_to_cart_order']]
        )

        # Apply Binning on 'days_since_prior_order'
        sampled_data['days_since_prior_order_temp'] = sampled_data['days_since_prior_order'] - 1
        sampled_data['days_since_prior_order_binned'] = pd.cut(
            sampled_data['days_since_prior_order_temp'],
            bins=[-1, 7, 15, 23, 31],
            labels=['0-7', '8-15', '16-23', '24-31'],
            right=True
        )
        sampled_data['days_since_prior_order_binned'] = sampled_data['days_since_prior_order_binned'].cat.add_categories('Unknown')
        sampled_data.loc[sampled_data['days_since_prior_order_binned'].isna(), 'days_since_prior_order_binned'] = 'Unknown'
        sampled_data.drop(columns=['days_since_prior_order_temp'], inplace=True)

        # Save the processed data to the output file
        output_folder = os.path.dirname(output_file)
        filename = os.path.basename(output_file)
        csv_save(sampled_data, output_folder, filename)
        print(f"Preprocessed data saved to {output_file}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
