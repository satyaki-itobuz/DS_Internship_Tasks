import pandas as pd

def feature_engineering(df):
    # Target encoding for 'department'
    department_target_encoding = df.groupby('department')['reordered'].mean()
    df['department_encoded'] = df['department'].map(department_target_encoding)

    # Target encoding for 'aisle'
    aisle_target_encoding = df.groupby('aisle')['reordered'].mean()
    df['aisle_encoded'] = df['aisle'].map(aisle_target_encoding)

    # Target encoding for 'product_name'
    product_name_target_encoding = df.groupby('product_name')['reordered'].mean()
    df['product_name_encoded'] = df['product_name'].map(product_name_target_encoding)

    # Drop original columns after encoding
    df = df.drop(columns=['product_name', 'department', 'aisle'])

    # Define time of day buckets (morning, afternoon, evening, night)
    def time_of_day(hour):
        if hour < 12:
            return 0  # Morning
        elif hour < 18:
            return 1  # Afternoon
        elif hour < 22:
            return 2  # Evening
        else:
            return 3  # Night

    df['time_of_day'] = df['order_hour_of_day'].apply(time_of_day)

    # Days Since Last Purchase (for each user-product pair)
    df['days_since_last_purchase'] = df.groupby(['user_id', 'product_id'])['days_since_prior_order'].shift(-1).fillna(0)

    # Number of orders per user
    df['user_order_count'] = df.groupby('user_id')['order_id'].transform('count')

    # Sum of items ordered per order (total items in each order)
    df['total_items_ordered'] = df.groupby('order_id')['add_to_cart_order'].transform('sum')

    # Total orders for each product
    df['product_popularity'] = df.groupby('product_id')['order_id'].transform('count')

    # Average order hour for each user
    df['avg_order_hour'] = df.groupby('user_id')['order_hour_of_day'].transform('mean')

    # Order Frequency (number of times a user has ordered a particular product)
    df['order_frequency'] = df.groupby(['user_id', 'product_id'])['order_id'].transform('count')

    # Product Affinity by Department (how much each user orders from each department)
    df['department_affinity'] = df.groupby(['user_id', 'department_id'])['order_id'].transform('count') / df.groupby('user_id')['order_id'].transform('count')

    # Time of Purchase Features: Preferred order hour and preferred order day of week
    df['preferred_order_hour'] = df.groupby('user_id')['order_hour_of_day'].transform('mean')
    df['preferred_order_dow'] = df.groupby('user_id')['order_dow'].transform('mean')

    # Average Reorder Rate for each user
    df['avg_reorder_rate'] = df.groupby(['user_id'])['reordered'].transform('mean')

    # Product Reorder Frequency across all users (how often each product gets reordered)
    df['product_reorder_frequency'] = df.groupby('product_id')['reordered'].transform('sum')

    # User's Reorder Rate for each product (how often a user reorders a particular product)
    df['user_product_reorder_rate'] = df.groupby(['user_id', 'product_id'])['reordered'].transform('mean')

    # Order Count for Product by User (how many times a user has ordered a particular product)
    df['user_product_order_count'] = df.groupby(['user_id', 'product_id'])['order_id'].transform('count')

    return df
