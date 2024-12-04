# ALL DATAFILE PATHS
ORDERS_CSV = '../data/orders.csv' 
PRODUCTS_CSV = '../data/products.csv'
ORDER_PRODUCTS_CSV = '../data/order_products.csv'
AISLE_CSV = '../data/aisles.csv'
PREPROCESSED_CSV = '../data/preprocessed_data.csv'

random_state = 42
target_column = 'reordered'
stratification_fraction = 0.1

FINAL_MODEL_PATH = './models/best_model.pkl'
EDA_PLOTS_LOCATION = "./plots/EDA/"
BUNDLES_LOCATION ="./results/product_bundles.csv"
CLUSTER_LOCATION ="./results/cluster_results.json"

HYPERPARAMETER_TUNING_DB_RESULTS = "./db/purchase_hp_tuning.db"