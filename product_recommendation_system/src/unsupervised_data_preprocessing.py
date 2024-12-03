import pandas as pd

def preprocess_unsupervised_data(df:pd.DataFrame):
    df.fillna(-1, inplace=True)
    df['days_since_prior_order'] = df['days_since_prior_order'] + 1
    return df


def stratified_sample(df:pd.DataFrame, stratify_col:str, frac: int) -> pd.DataFrame:
    
    stratified_df = df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))

    stratified_df = stratified_df.reset_index(drop=True)
    
    return stratified_df

def aggregate_user_data(df:pd.DataFrame):
    """
    Aggregates the data to create a single row per user with summarized features for clustering.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing user purchase data.

    Returns:
    - pd.DataFrame: Aggregated user data with one row per user.
    """
    
    # 1. Total Orders per User (count of unique orders)
    total_orders = df.groupby('user_id')['order_id'].nunique().reset_index()
    total_orders.rename(columns={'order_id': 'total_orders'}, inplace=True)

    # 2. Days Since Last Order (recency of last order)
    days_since_last_order = df.groupby('user_id')['days_since_prior_order'].last().reset_index()
    days_since_last_order.rename(columns={'days_since_prior_order': 'days_since_last_order'}, inplace=True)

    # 3. Average Days Between Orders (order frequency)
    avg_order_frequency = df.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    avg_order_frequency.rename(columns={'days_since_prior_order': 'avg_days_between_orders'}, inplace=True)

    # 4. Unique Products Purchased (number of different products bought)
    unique_products = df.groupby('user_id')['product_id'].nunique().reset_index()
    unique_products.rename(columns={'product_id': 'unique_products'}, inplace=True)

    # 5. Most Frequent Aisle (mode of aisle_id)
    most_frequent_aisle = df.groupby('user_id')['aisle_id'].agg(lambda x: x.mode()[0]).reset_index()
    most_frequent_aisle.rename(columns={'aisle_id': 'most_frequent_aisle'}, inplace=True)

    # 6. Most Frequent Department (mode of department_id)
    most_frequent_dept = df.groupby('user_id')['department_id'].agg(lambda x: x.mode()[0]).reset_index()
    most_frequent_dept.rename(columns={'department_id': 'most_frequent_dept'}, inplace=True)

    # 7. Basket Size (average number of items per order)
    basket_size = df.groupby('user_id')['order_id'].count().reset_index()
    basket_size.rename(columns={'order_id': 'basket_size'}, inplace=True)

    # 8. Repeat Purchases (percentage of reordered items)
    repeat_purchases = df.groupby('user_id')['product_id'].apply(lambda x: (x.duplicated().sum() / len(x)) * 100).reset_index()
    repeat_purchases.rename(columns={'product_id': 'repeat_purchase_percentage'}, inplace=True)

    # 9. Average Hour of Day for Purchases (when users tend to purchase)
    avg_purchase_time = df.groupby('user_id')['order_hour_of_day'].mean().reset_index()
    avg_purchase_time.rename(columns={'order_hour_of_day': 'avg_purchase_hour'}, inplace=True)

    # 10. Weekend Shopper (percentage of purchases made on weekends)
    df['is_weekend'] = df['order_dow'].apply(lambda x: 1 if x in [0, 6] else 0)
    weekend_shopper = df.groupby('user_id')['is_weekend'].mean().reset_index()
    weekend_shopper.rename(columns={'is_weekend': 'weekend_shopper_percentage'}, inplace=True)

    # 11. Last Purchased Aisle and Department (based on last order)
    last_purchased_aisle = df.groupby('user_id')['aisle_id'].last().reset_index()
    last_purchased_aisle.rename(columns={'aisle_id': 'last_purchased_aisle'}, inplace=True)

    last_purchased_dept = df.groupby('user_id')['department_id'].last().reset_index()
    last_purchased_dept.rename(columns={'department_id': 'last_purchased_dept'}, inplace=True)

    # Merging all features into a single dataframe (join on user_id)
    df_aggregated = total_orders.merge(days_since_last_order, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(avg_order_frequency, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(unique_products, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(most_frequent_aisle, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(most_frequent_dept, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(basket_size, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(repeat_purchases, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(avg_purchase_time, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(weekend_shopper, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(last_purchased_aisle, on='user_id', how='left')
    df_aggregated = df_aggregated.merge(last_purchased_dept, on='user_id', how='left')

    return df_aggregated