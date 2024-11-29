import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectKBest, f_classif

def feature_importance(X, y, log_reg, alpha=0.01, top_k=10):
    # Logistic Regression Coefficients (Absolute Value)
    importance = np.abs(log_reg.coef_[0])
    feature_names = X.columns
    
    # Create a DataFrame to store feature names and their corresponding importance
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    
    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Features')
    plt.title('Feature Importance Based on Absolute Coefficients (Logistic Regression)')
    plt.gca().invert_yaxis()  # To have the most important feature on top
    plt.show()
    
    # Lasso Regularization (L1 Regularization) for Feature Selection
    lasso = Lasso(alpha=alpha)  # Lasso regularization
    lasso.fit(X, y)
    lasso_selected_features = X.columns[lasso.coef_ != 0]
    print(f"Selected features using Lasso regularization: {lasso_selected_features}")
    
    # Univariate Statistical Tests (ANOVA F-test) for Feature Selection
    selector = SelectKBest(f_classif, k=top_k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Top {top_k} features based on ANOVA F-test: {selected_features}")

