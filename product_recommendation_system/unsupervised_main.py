import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.unsupervised_data_preprocessing import preprocess_unsupervised_data, stratified_sample, aggregate_user_data
from src.unsupervised_user_segmentation import user_segmentation_using_clustering_with_GMM, user_segmentation_using_clustering_with_kmeans
from config import order_products, order, products
from datetime import datetime
import logging
import os


# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f'unsupervised_run_{timestamp}.log')# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)


try:
    # Load and preprocess data
    df_products = pd.read_csv(products)
    df_ordered_products = pd.read_csv(order_products)
    df_orders = pd.read_csv(order)

except Exception as FailedToReadDataError:
    logging.error('can not read data')



try:
    # merge data
    df = pd.merge(df_orders, df_ordered_products, on='order_id', how='inner')
    df.drop(columns=['eval_set', 'order_number', 'add_to_cart_order','reordered'], inplace=True, axis=1)
    df = pd.merge(df, df_products, on='product_id', how='inner')

except Exception as MergeError:
    logging.error('can not merge data')



try:
    # preprocess data
    df = preprocess_unsupervised_data(df)
except Exception as PreprocessError:
    logging.error(msg=f'some error occured while preprocessing the data : {repr(PreprocessError)}')


try:
    # stratify data
    stratified_df = stratified_sample(df, 'user_id', frac=0.3)
except Exception as StratifingError:
    logging.error(msg=f'some error occured while stratifing the data : {repr(StratifingError)}')


try:
    # Creating unique user profiles
    user_profiles = aggregate_user_data(stratified_df)
except Exception as UserProfileError:
    logging.error(msg=f'some error occured while generating user profiles : {repr(UserProfileError)}')


try:
    # scale features
    features = user_profiles.drop(columns=['user_id'], axis=1)
    features.dropna(inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # user segmentation
    user_segmentation_using_clustering_with_kmeans(features, scaled_features,user_profiles)
    user_segmentation_using_clustering_with_GMM(features, scaled_features,user_profiles)

except Exception as ClusteringError:
    logging.error(msg=f'some error occured while segmenting users using clustering techniques : {repr(ClusteringError)}')


# final results : eda on clustering


   



