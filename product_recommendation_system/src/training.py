
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import logging
from config import *

logger = logging.getLogger(__name__)

def model_final(X, y):
    """
    Trains different models and evaluates them using various metrics.
    
    Args:
    X : pandas DataFrame or numpy array
        Feature matrix.
    y : pandas Series or numpy array
        Target variable.
    sample_fraction : float, optional (default=0.4)
        Fraction of the dataset to sample for training (ignored as test_size is fixed).

    Returns:
    best_model : scikit-learn model object
        The model with the best F1-score.
    best_model_name : str
        Name of the best model.
    """
    
    # Split the dataset into training and test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as Data_not_splitted:
        logger.error(f"Error during data splitting: {str(Data_not_splitted)}")
        return
    
    # Dictionary of models to evaluate
    models = {
        'Logistic Regression (L1 Regularization)': LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1'),
    }
    
    # Initialize variables to track the best model
    best_model = None
    best_model_name = ""
    best_f1_score_value = 0

    # Loop through each model to train and evaluate
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
        except Exception as training_error:
            logger.error(f"Error while training the model : {str(training_error)}")

        try: 
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception as prediction_error:
            logger.error(f"Error while drawing prediction: {str(prediction_error)}")
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1_score_value = f1_score(y_test, y_pred)
    
        logger.info(f"Model: {model_name}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"F1-Score: {f1_score_value:.4f}")
        logger.info("-" * 40)
        
        # Select the best model based on F1-Score
        if f1_score_value > best_f1_score_value:
            best_f1_score_value = f1_score_value
            best_model = model
            best_model_name = model_name
    
    # Print the best model information
    print(f"\nBest Model: {best_model_name} with F1 score of {best_f1_score_value:.4f}")
    
    # Save the best model using pickle
    try:
        with open(final_model_path, 'wb') as file:
            pickle.dump(best_model, file)
            logger.info(f"The best model has been saved as {final_model_path}.")
    except Exception as Model_not_saved:
        logger.error(f"Error while saving the model {str(Model_not_saved)}")

    return best_model, best_model_name
