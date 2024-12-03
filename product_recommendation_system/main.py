from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_selection import model_selection,evaluate_model
from src.training import model_final
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import logging
import time
from datetime import datetime
import os
from config import *


# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join('log', f'supervised_run_{timestamp}.log')# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)


def main(full_path):
    try:
        X, y = load_and_preprocess_data(full_path)
        logging.info(f"Data loaded and preprocessed successfully. Features shape: {X.shape}, Labels shape: {y.shape}")
    except Exception as Data_not_loaded:
        logging.error(f"Error during data loading and preprocessing: {str(Data_not_loaded)}")
        return

    # Models for selection
    models = {
    'Logistic Regression (L2 Regularization)': LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2'),
    'Logistic Regression (L1 Regularization)': LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1'),
    'Logistic Regression (ElasticNet Regularization)': LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5),
    'Naive Bayes': GaussianNB(),
    'Decision Stump': DecisionTreeClassifier(max_depth=1),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'SGD Classifier (Log Loss)': SGDClassifier(loss='log_loss', max_iter=1000),
    'SGD Classifier (Hinge Loss)': SGDClassifier(loss='hinge', max_iter=1000),
    'Random Forest (Shallow Depth)': RandomForestClassifier(n_estimators=10, max_depth=3),
    'Ridge Classifier': RidgeClassifier(),
    'Linear SVC': LinearSVC(max_iter=1000),
    'QDA': QuadraticDiscriminantAnalysis(),
}
    try:
        best_model, best_model_name = model_selection(X, y, models, scoring='accuracy', k=5)
        logging.info(f"Best model selected: {best_model_name}")
    except Exception as Model_selection_failed:
        logging.error(f"Error during model selection: {str(Model_selection_failed)}")
        return

    try:
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        logging.info("Data split into training and testing sets.")
    except Exception as Data_not_loaded:
        logging.error(f"Error during data split: {str(Data_not_loaded)}")
        return
    
    # After cross validation for final evaluation
    final_models = {
    'Logistic Regression (L1 Regularization)': LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1'),
    'Decision Stump': DecisionTreeClassifier(max_depth=1),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Random Forest (Shallow Depth)': RandomForestClassifier(n_estimators=10, max_depth=3),
    'QDA': QuadraticDiscriminantAnalysis(),
    }
    
    try:
        best_final_model, best_final_model_name = evaluate_model(X_train, X_test, y_train, y_test, final_models)
        logging.info(f"Best final model evaluated: {best_final_model_name}")
    except Exception as evaluation_error:
        logging.error(f"Error during final model evaluation: {str(evaluation_error)}")
        return

    try:
        best_model, best_final_model_name = model_final(X, y)
        logging.info(f"Best final model trained: {best_final_model_name}")
    except Exception as model_training_failed:
        logging.error(f"Error during final model training: {str(model_training_failed)}")
        return

if __name__ == "__main__":
    try:
        main(full_path)
    except Exception as main_not_executed:
        logging.error(f"Error in executing main: {str(main_not_executed)}")
