# required dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import f1_score
import logging, sys
import optuna


logger = logging.getLogger(__name__)

# Loading the preprocessed dataset
def load_dataset(path:str) -> pd.DataFrame:
    """
    Loads the dataset
    Args:
    - path (str): path to the dataset
    Returns:
    - data (dataframe): the loaded the dataset
    """
    try:
        data = pd.read_csv(filepath_or_buffer=path)
        data = data.drop(columns=(["Unnamed: 0","order_id","product_id","user_id","days_since_prior_order_binned"]))
        logger.info
        return data
    except Exception as error:
        logger.error(error)
        return repr(error)

# Splitting the data into training and testing set
def data_split(data:pd.DataFrame):
    """
    Splits the data into training and testing set.
    Args:
    - data (dataframe): dataset
    
    Returns:
    - X_train (array-like): training set of feature matrix.
    - X_test (array-like): testing set of feature matrix.
    - y_train (array-like): training set of target vector.
    - y_test (array-like): testing set of target vector.
    """
    X = data.drop(columns="reordered")
    y = data["reordered"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test

# ObJective Function
def objective_function(trial:optuna.trial,X_train, X_test, y_train, y_test) -> float:
    '''
    Objective Function whcih computes f1_score of the trained model.
    Args:
    - trail : optuna trail which will be run.

    Returns:
    - f1_score (float): evaluates the f1_score of the trained model.
    '''
    # Hyperparameter Search Space
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-4, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-4, 10.0),
        "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split", 1e-4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }
    logger.info(param_grid)
    model = lightgbm.LGBMClassifier(**param_grid, n_jobs=1, learning_rate=0.01, random_state=42)
    model.fit(X_train.fillna(0),y_train)
    y_pred = model.predict(X_test.fillna(0))
    
    return f1_score(y_test,y_pred)

# Create Optuna study
def create_load_study() -> optuna.study:
    '''
    Creates a study if no study previously created or loads one for hyperparameter tuning
    Returns:
    - study (optuna.study) - study where all trial are to be stored
    '''
    try:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "purchase_hp_tuning"
        storage_name = "sqlite:///results/{}.db".format(study_name)

        study = optuna.create_study(study_name = study_name, storage=storage_name, load_if_exists=True, direction='maximize')
        return study
    except Exception as error:
        logger.error(error)

# Hyperparameter Tuning using optuna
def hyperparameter_tuning(ntrial : int, study:optuna.study,X_train, X_test, y_train, y_test) -> optuna.study:
    '''
    Performs Hyperparameter tuning on Model using optuna
    Requires a defined objective function to maximize
    Args:
    - ntrial (int) - number of trials to be conducted.
    - study (optuna.study) - study where all trial are stored

    Returns:
    - study - study in which trials are done
    '''
    try:
        logger.info(f"Starting hypermeter tuning with {ntrial} trials.")
        study.optimize(lambda trial: objective_function(trial, X_train, X_test, y_train, y_test), n_trials=ntrial, n_jobs=2)
        logger.info("Hyperparameter tuning completed successfully.")
        return study
    except Exception as error:
        logger.error(error)

# Best Trial in the study
def best_trial(study:optuna.study):
    '''
    Returns parameters and f1_score of the best trial in the given study
    Args:
    - study (optuna.study)
    Returrns:
    - best_params (dict) - A dictionnary of parameters which give the best score
    - best_value (float) - The best score within the entire study
    '''
    best_params =  study.best_params
    best_value =  study.best_value
    return (best_params, best_value)