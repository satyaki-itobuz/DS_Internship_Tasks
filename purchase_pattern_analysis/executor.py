# DEPENDENCIES
import json
import logging
import pandas as pd
from datetime import datetime
import os 
from config import ORDERS_CSV
from config import PRODUCTS_CSV
from config import ORDER_PRODUCTS_CSV
from config import AISLE_CSV
# from src.market_basket import market_basket_analysis
from src.customer_segmentation import customer_segmentation_analysis

# Setup logging 
# Create log filename with timestamp
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f'unsupervised_app_{timestamp}.log')# Configure logging
logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'),  # Log to file
                              logging.StreamHandler()]  # Also log to console
                   )


# READ / LOAD DATA
orders = pd.read_csv(filepath_or_buffer = ORDERS_CSV)
products = pd.read_csv(filepath_or_buffer = PRODUCTS_CSV)
order_products = pd.read_csv(filepath_or_buffer = ORDER_PRODUCTS_CSV)
aisles = pd.read_csv(filepath_or_buffer = AISLE_CSV)
logging.info('All data Loaded successfulyy')

# Merge orders and order-products on 'order_id'
merged_df = pd.merge(order_products, orders, on='order_id', how='inner')
logging.info("order_products and order data merged successfully")
# Merge the result with the products dataset on 'product_id'
final_merged_df = pd.merge(merged_df, products, on='product_id', how='inner')

# View the resulting merged dataset
# print(final_merged_df.head())
#logging.info(final_merged_df.head())


# print(orders.head())
# print(aisles.head())
# product_bundle_result = market_basket_analysis(orders_data   = order_prodcuts,
#                                                products_data = prodcuts)

# print(product_bundle_result)

# product_bundle_result.to_csv(path_or_buf = './results/product_bundles.csv',
#                              index        = None)

customer_segmentation_result = customer_segmentation_analysis(merged_data = final_merged_df,
                                                              aisle_data  = aisles)


with open('./results/cluster_results.json', 'w') as fp: 
    # Make dictionary items JSON-serializable
    json_data = {key : value.to_json() for key, value in customer_segmentation_result.items()}
    json.dump(json_data, fp, indent = 4)
fp.close()
