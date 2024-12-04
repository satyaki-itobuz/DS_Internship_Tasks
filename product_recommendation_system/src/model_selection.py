import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from config import *

logger=logging.getLogger(__name__)

def model_selection(X, y, models, scoring='accuracy', k=10, sample_fraction=0.3):
    """
    Perform model selection on a stratified 30% sample of the dataset using k-fold cross-validation.
    Args:
    - X (array-like): Feature matrix.
    - y (array-like): Target vector.
    - models (dict): Dictionary of models to evaluate.
    - scoring (str): Scoring metric for evaluation. Default is 'accuracy'.
    - k (int): Number of folds for cross-validation. Default is 5.
    - sample_fraction (float): Fraction of the dataset to sample. Default is 0.3 (30% sample).
    
    Returns:
    Returning a tuple which consist of
    - best_model (model): The best performing model based on average k-fold score.
    - best_model_name (str): The name of the best performing model.
    """
    try:
        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=1-sample_fraction, stratify=y)
    except Exception as splitting_failed:
        logger.error(f"Error while splitting the data : {str(splitting_failed)}")
        return
    cv = StratifiedKFold(n_splits=k, shuffle=True)
    
    best_model = None
    best_model_name = None
    best_score = -np.inf
    try:
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
        
            start_time = time.time()
            cv_scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring=scoring)
            avg_score = np.mean(cv_scores)  # Get average score across all folds
            end_time = time.time()
        
            logger.info(f"{model_name}: Average {scoring} = {avg_score:.4f} (Time taken: {end_time - start_time:.2f} seconds)")
        
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_model_name = model_name
    
        logger.info(f"\nBest Model: {best_model_name} with Average {scoring}: {best_score:.4f}")
    
        return best_model, best_model_name
    except Exception as model_selection_error:
        logger.error(f"Error while selcting the model {str(model_selection_error)}")


def evaluate_model(X_train, X_test, y_train, y_test, models):
    """
    Train models and evaluate their performance.
    Args:
    - X_train, X_test: Training and testing feature matrices.
    - y_train, y_test: Training and testing target vectors.
    - models (dict): Dictionary of models to evaluate.
    
    Returns:
    - best_model (model): The best performing model based on F1 score.
    - best_model_name (str): The name of the best performing model.
    """
    best_model = None
    best_model_name = ""
    best_f1_score_value = 0
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
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
        
            if f1_score_value > best_f1_score_value:
                best_f1_score_value = f1_score_value
                best_model = model
                best_model_name = model_name
    except Exception as Evaluation_failed:
        logger.error(f"Error while evaluating the model {str(Evaluation_failed)}")
    
    logger.info(f"\nBest Model: {best_model_name} with F1 score of {best_f1_score_value:.4f}")
    
    return best_model, best_model_name
