import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)

def feature_engineering(df):
    try:
        # Log the start of the process
        logger.info("Starting the feature engineering process.")

        # Time of Day Buckets
        logger.info("Creating time_of_day feature based on order_hour_of_day.")
        df['time_of_day'] = df['order_hour_of_day'].apply(
            lambda hour: 0 if hour < 12 else (1 if hour < 18 else (2 if hour < 22 else 3))
        )

        # Days Since Last Purchase
        logger.info("Calculating days_since_last_purchase.")
        df['days_since_last_purchase'] = df.groupby(['user_id', 'product_id'])['days_since_prior_order'].shift(-1).fillna(0)

        # Feature aggregations
        logger.info("Creating various aggregation features.")
        df['user_order_count'] = df.groupby('user_id')['order_id'].transform('count')
        df['total_items_ordered'] = df.groupby('order_id')['add_to_cart_order'].transform('sum')
        df['product_popularity'] = df.groupby('product_id')['order_id'].transform('count')
        df['avg_order_hour'] = df.groupby('user_id')['order_hour_of_day'].transform('mean')
        df['order_frequency'] = df.groupby(['user_id', 'product_id'])['order_id'].transform('count')
        df['department_affinity'] = (
            df.groupby(['user_id', 'department_id'])['order_id'].transform('count') /
            df.groupby('user_id')['order_id'].transform('count')
        )
        df['preferred_order_hour'] = df.groupby('user_id')['order_hour_of_day'].transform('mean')
        df['preferred_order_dow'] = df.groupby('user_id')['order_dow'].transform('mean')
        df['avg_reorder_rate'] = df.groupby(['user_id'])['reordered'].transform('mean')
        df['product_reorder_frequency'] = df.groupby('product_id')['reordered'].transform('sum')
        df['user_product_reorder_rate'] = df.groupby(['user_id', 'product_id'])['reordered'].transform('mean')
        df['user_product_order_count'] = df.groupby(['user_id', 'product_id'])['order_id'].transform('count')

        # Target Encoding
        logger.info("Performing target encoding on categorical features.")
        columns_to_encode = ['aisle', 'product_name', 'department']
        target_column = 'reordered'
        for col in columns_to_encode:
            df[col] = df[col].map(df.groupby(col)[target_column].mean())

        # Correlation Matrix and Feature Reduction
        logger.info("Computing correlation matrix and dropping highly correlated features.")
        correlation_matrix = df.corr()
        columns_to_drop = [('order_hour_of_day', 'time_of_day'),
                           ('product_popularity', 'product_reorder_frequency'),
                           ('avg_order_hour', 'preferred_order_hour'),
                           ('order_frequency', 'user_product_order_count')]
        for col1, col2 in columns_to_drop:
            df.drop(columns=[col2], inplace=True)

        # Feature Scaling
        logger.info("Scaling selected features using MinMaxScaler.")
        scaler = MinMaxScaler()
        columns_to_scale = [
            'product_id', 'order_id', 'add_to_cart_order', 'reordered', 'user_id',
            'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
            'days_since_last_purchase', 'user_order_count', 'total_items_ordered',
            'product_popularity', 'avg_order_hour', 'order_frequency', 'department_affinity',
            'preferred_order_dow', 'avg_reorder_rate', 'user_product_reorder_rate'
        ]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        # Train-Test Split
        logger.info("Splitting the data into training, testing, and validation sets.")
        X = df.drop(columns=['reordered'])
        y = df['reordered']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test_final, X_val, y_test_final, y_val = train_test_split(X_test, y_test, test_size=0.15, random_state=42)

        # Logistic Regression
        logger.info("Training Logistic Regression model.")
        log_reg = LogisticRegression(max_iter=500, solver='lbfgs')
        log_reg.fit(X_train, y_train)

        # Model Evaluation
        logger.info("Evaluating the model on the test set.")
        y_pred = log_reg.predict(X_test_final)
        accuracy = accuracy_score(y_test_final, y_pred)
        f1 = f1_score(y_test_final, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Model F1 Score: {f1:.4f}")

        # Feature Importance
        logger.info("Computing feature importance using Logistic Regression coefficients.")
        importance = np.abs(log_reg.coef_[0])
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

        # Saving the Model
        logger.info("Saving the trained model.")
        with open('model.pkl', 'wb') as f:
            pickle.dump(log_reg, f)

        # Visualization of Confusion Matrix
        logger.info("Generating confusion matrix heatmap.")
        cm = confusion_matrix(y_test_final, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        logger.info("Feature engineering process completed successfully.")
        return df

    except Exception as e:
        logger.error(f"An error occurred during feature engineering: {e}")
        raise
