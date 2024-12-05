# ALL DATAFILE PATHS
ORDERS_CSV                       = "../data/orders.csv"
PRODUCTS_CSV                     = "../data/products.csv"
ORDER_PRODUCTS_CSV               = "../data/order_products.csv"
AISLE_CSV                        = "../data/aisles.csv"
ENCODED_MERGED_DATA              = "../data/encoded_data.csv"
PROCESSED_ORDERS                 = "../data/orders_processed.csv"
PREPROCESSED_CSV                 = "../data/ppa_preprocessed_data.csv"

# GLOBAL VARIABLES
RANDOM_STATE                     = 42
TARGET_COLUMN                    = "reordered"
STRATIFICATION_FRACTION          = 0.1

# RESULTS PATHS
FINAL_MODEL_PATH                 = "./models/best_model.pkl"
EDA_PLOTS_LOCATION               = "./plots/EDA/"
BUNDLES_LOCATION                 = "./results/product_bundles.csv"
CLUSTER_LOCATION                 = "./results/cluster_results.json"

HYPERPARAMETER_TUNING_DB_RESULTS = "./db/purchase_hp_tuning.db"