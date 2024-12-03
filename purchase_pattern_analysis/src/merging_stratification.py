import os
import pandas as pd

def load_datasets(data_folder):
    """
    Load datasets from the specified data folder.
    """
    try:
        order_products = pd.read_csv(os.path.join(data_folder, "order_products.csv"))
        orders = pd.read_csv(os.path.join(data_folder, "orders.csv"))
        products = pd.read_csv(os.path.join(data_folder, "products.csv"))
        aisles = pd.read_csv(os.path.join(data_folder, "aisles.csv"))
        departments = pd.read_csv(os.path.join(data_folder, "departments.csv"))
        return {
            "order_products": order_products,
            "orders": orders,
            "products": products,
            "aisles": aisles,
            "departments": departments
        }
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None

def merge_datasets(datasets):
    """
    Merge the loaded datasets to create a comprehensive DataFrame.
    
    Parameters:
    - datasets (dict): Dictionary containing loaded DataFrames.
    
    Returns:
    - Merged DataFrame.
    """
    try:
        # Merge aisles and departments with products
        products = datasets["products"].merge(datasets["aisles"], on="aisle_id", how="left")
        products = products.merge(datasets["departments"], on="department_id", how="left")
        
        # Drop unnecessary columns from order_products
        order_products = datasets["order_products"].drop(
            columns=["product_name", "aisle_id", "department_id"], errors="ignore"
        )
        
        # Merge order_products with enriched products
        order_products = order_products.merge(products, on="product_id", how="left")
        
        # Select necessary columns from orders
        req_orders = datasets["orders"][
            ["order_id", "user_id", "eval_set", "order_number", 
             "order_dow", "order_hour_of_day", "days_since_prior_order"]
        ]
        
        # Merge order_products with orders
        merged_df = order_products.merge(req_orders, on="order_id", how="left")
        
        return merged_df
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def treat_missing_values(df):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to process.
    
    Returns:
    - DataFrame with treated missing values.
    """
    try:
        df["days_since_prior_order"] = df["days_since_prior_order"].fillna(-1)
        df["days_since_prior_order"] = df["days_since_prior_order"] + 1
        return df
    except Exception as e:
        print(f"Error treating missing values: {e}")
        return df

def stratified_sample(df, stratify_col, frac):
    """
    Perform stratified sampling on the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to sample from.
    - stratify_col (str): Column for stratification.
    - frac (float): Fraction of rows to sample.
    
    Returns:
    - Sampled DataFrame.
    """
    try:
        stratified_df = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42)
        )
        return stratified_df.reset_index(drop=True)
    except Exception as e:
        print(f"Error performing stratified sampling: {e}")
        return df

def save_to_csv(df, output_folder, filename):
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