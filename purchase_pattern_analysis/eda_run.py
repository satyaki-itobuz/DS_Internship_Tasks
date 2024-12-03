import pandas as pd
import json
from src.config import ORDERS
from src.config import ORDER_PRODUCTS
from src.config import PRODUCTS
from src.config import AISLES
from src.config import DATA
from src.config import ORDERS_UNCLEAN
from src.Exploratory_Data_Analysis import *
from src.Data_Description import *
from src.Data_Preprocessing import *

#load/read data
orders = pd.read_csv(filepath_or_buffer= ORDERS, index_col = None)
order_products = pd.read_csv(filepath_or_buffer= ORDER_PRODUCTS, index_col = None)
aisles = pd.read_csv(filepath_or_buffer= AISLES, index_col = None)
products = pd.read_csv(filepath_or_buffer= PRODUCTS, index_col = None)
data = pd.read_csv(filepath_or_buffer= DATA, index_col = None)
unclean_orders = pd.read_csv(filepath_or_buffer= ORDERS_UNCLEAN, index_col = None)

print(merge_datasets(order_products, products, unclean_orders, aisles))


