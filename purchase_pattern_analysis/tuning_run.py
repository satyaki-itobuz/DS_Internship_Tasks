import logging
from src.hyperparameter_tuning import load_dataset, data_split, objective_function
from src.hyperparameter_tuning import create_load_study, hyperparameter_tuning, best_trial

LOG_FILE = "results/tuning_run.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w" 
)


logging.info("Starting the hyperparameter tuning process.")
    
try:
    data = load_dataset()
    logging.info("Dataset loaded successfully.")
        
    X_train, X_test, y_train, y_test = data_split(data)
    logging.info("Data split into training and testing sets.")
        
    study = create_load_study()
    logging.info("Study created/loaded.")
        
    study = hyperparameter_tuning(1)
    logging.info("Hyperparameter tuning completed.")
        
    x, y = best_trial(study)
    logging.info(f"Best parameters = {x}; Best score = {y}")
except Exception as error:
    logging.error(f"An error occurred: {error}", exc_info=True)


