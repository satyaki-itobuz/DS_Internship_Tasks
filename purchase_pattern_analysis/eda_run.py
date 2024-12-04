# DEPENDENCIES
import os
import logging
import pandas as pd
from config import AISLE_CSV
from config import ORDERS_CSV
from datetime import datetime
from config import PRODUCTS_CSV
from config import PROCESSED_ORDERS
from config import ORDER_PRODUCTS_CSV
from config import ENCODED_MERGED_DATA
from src.data_preprocessing import drop_columns
from src.exploratory_data_analysis import eda_workflow


# Create log filename with timestamp
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f'eda_run_{timestamp}.log')# Configure logging
logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_filename, encoding='utf-8'),  # Log to file
                              logging.StreamHandler()]  # Also log to console
                   )

#load/read data
orders              = pd.read_csv(filepath_or_buffer = ORDERS_CSV, 
                                  index_col          = None)

order_products      = pd.read_csv(filepath_or_buffer = ORDER_PRODUCTS_CSV, 
                                  index_col          = None)

aisles              = pd.read_csv(filepath_or_buffer = AISLE_CSV, 
                                  index_col          = None)

products            = pd.read_csv(filepath_or_buffer = PRODUCTS_CSV, 
                                  index_col          = None)

processed_orders    = pd.read_csv(filepath_or_buffer = PROCESSED_ORDERS, 
                                  index_col          = None)

encoded_merged_data = pd.read_csv(filepath_or_buffer = ENCODED_MERGED_DATA, 
                                  index_col          = None)
logging.info("All Data Loaded.")


drop_columns(orders, 'eval_set')
eda_workflow(orders_data         = processed_orders, 
             order_products_data = order_products, 
             aisles_data         = aisles, 
             products_data       = products, 
             merged_data         = encoded_merged_data)






