# DATA PATHS
ORDER                               = "../data/orders.csv"
AISLES                              = "../data/aisles.csv"
ORDER_PRODUCTS                      = "../data/order_products.csv"
PRODUCTS                            = "../data/products.csv"
DEPARTMENTS                         = "../data/departments.csv"
STRATIFIED_DATA                     = "../data/stratified_data.csv"

# SOME GLOBAL PARAMETERS
STRATIFICATION_FRACTION             = 0.1
RANDOM_STATE                        = 42
TARGET_COLUMN                       = 'reordered'

# RESULTS LOCATIONS FOR ALL TASKS
EDA_TEST_RESULTS                    = './results/EDA/test_results.json'
EDA_PLOTS_LOCATION                  = "./plots/EDA/"
FINAL_MODEL_PATH                    = './models/best_model.pkl'
HYPERPARAMETER_TUNING_TOTAL_RESULTS = './results/hyperparameter_tuning_total_rersults.json'
HYPERPARAMETER_TUNING_TOP5_RESULTS  = './results/hyperparameter_tuning_top5_rersults.csv'
