
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict on test data
        y_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for ROC-AUC
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1_score_value = f1_score(y_test, y_pred)
        
        # Print the evaluation metrics for each model
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1-Score: {f1_score_value:.4f}")
        print("-" * 40)
        
        # Select the best model based on F1-Score
        if f1_score_value > best_f1_score_value:
            best_f1_score_value = f1_score_value
            best_model = model
            best_model_name = model_name
    
    # Print the best model information
    print(f"\nBest Model: {best_model_name} with F1 score of {best_f1_score_value:.4f}")
    
    # Save the best model using pickle
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
        print(f"The best model has been saved as 'best_model.pkl'.")

    return best_model, best_model_name
