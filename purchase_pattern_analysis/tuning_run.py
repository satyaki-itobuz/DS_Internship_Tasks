import os
import logging
from datetime import datetime
from config import PREPROCESSED_CSV
from src.hyperparameter_tuning import load_dataset, data_split, objective_function
from src.hyperparameter_tuning import create_load_study, hyperparameter_tuning, best_trial

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"log/tuning_run_{current_time}.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ],
)

logging.info("Starting the hyperparameter tuning process.")
    
try:
    data = load_dataset(PREPROCESSED_CSV)
    logging.info("Dataset loaded successfully.")
        
    X_train, X_test, y_train, y_test = data_split(data)
    logging.info("Data split into training and testing sets.")
        
    study = create_load_study()
    logging.info("Study created/loaded.")
        
    study = hyperparameter_tuning(1,study,X_train, X_test, y_train, y_test)
    logging.info("Hyperparameter tuning completed.")
        
    x, y = best_trial(study)
    logging.info(f"Best parameters = {x}; Best score = {y}")
except Exception as error:
    logging.error(f"An error occurred: {error}", exc_info=True)


