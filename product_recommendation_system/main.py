from src.data_preprocessing import load_and_preprocess_data, split_data
from src.model_selection import model_selection,evaluate_model
from src.training import model_final
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import time

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('../project_1_dataset/filename.csv')

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

    # Step 1: Model Selection
    best_model, best_model_name = model_selection(X, y, models, scoring='accuracy', k=5)
    
    # Step 2: Final Model Evaluation
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # After cross validation for final evaluation
    final_models = {
    'Logistic Regression (L1 Regularization)': LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1'),
    'Decision Stump': DecisionTreeClassifier(max_depth=1),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Random Forest (Shallow Depth)': RandomForestClassifier(n_estimators=10, max_depth=3),
    'QDA': QuadraticDiscriminantAnalysis(),
    }
    
    best_final_model, best_final_model_name = evaluate_model(X_train, X_test, y_train, y_test, final_models)

    # Training the final best fir model 
    best_model,best_final_model_name=model_final(X,y)

if __name__ == "__main__":
    main()
