import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_datasets(file_path):
    """Load the dataset from the specified CSV file path."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")


def save_pickle(data, output_folder, filename):
    """Save data to a pickle file."""
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, filename), 'wb') as f:
        pickle.dump(data, f)


def save_csv(df, output_folder, filename):
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


def average_days_between_purchases(data, results_folder):
    """Calculate average days between purchases for each user-product pair."""
    feature = data.groupby(['user_id', 'product_id'])['days_since_prior_order'].mean().reset_index()
    feature.rename(columns={'days_since_prior_order': 'average_days_between_purchases'}, inplace=True)
    save_pickle(feature, results_folder, 'average_days_between_purchases.pkl')
    return feature


def product_purchase_frequency(data, results_folder):
    """Calculate product purchase frequency for each user-product pair."""
    feature = data.groupby(['user_id', 'product_id']).size().reset_index(name='product_purchase_frequency')
    save_pickle(feature, results_folder, 'product_purchase_frequency.pkl')
    return feature


def purchase_stats(data, results_folder):
    """Calculate total purchases and interval standard deviation for each user-product pair."""
    feature = data.groupby(['user_id', 'product_id']).agg(
        total_purchases=('order_id', 'count'),
        interval_std_dev=('days_since_prior_order', 'std')
    ).reset_index()
    save_pickle(feature, results_folder, 'purchase_stats.pkl')
    return feature


def product_reorder_rate(data, results_folder):
    """Calculate the reorder rate for each product."""
    feature = data.groupby('product_id').agg(
        total_orders=('order_id', 'count'),
        total_reorders=('reordered', 'sum')
    ).reset_index()
    feature['product_reorder_rate'] = feature['total_reorders'] / feature['total_orders']
    save_pickle(feature[['product_id', 'product_reorder_rate']], results_folder, 'product_reorder_rate.pkl')
    return feature[['product_id', 'product_reorder_rate']]


def users_general_reorder_rate(data, results_folder):
    """Calculate the general reorder rate for each user."""
    feature = data.groupby('user_id').agg(
        total_items=('order_id', 'count'),
        reordered_items=('reordered', 'sum')
    ).reset_index()
    feature['users_general_reorder_rate'] = feature['reordered_items'] / feature['total_items']
    save_pickle(feature[['user_id', 'users_general_reorder_rate']], results_folder, 'users_general_reorder_rate.pkl')
    return feature[['user_id', 'users_general_reorder_rate']]


def avg_add_to_cart_order(data, results_folder):
    """Calculate the average position of each product in the cart."""
    feature = data.groupby('product_id')['add_to_cart_order'].mean().reset_index()
    feature['avg_add_to_cart_order'] = feature['add_to_cart_order']
    save_pickle(feature, results_folder, 'avg_add_to_cart_order.pkl')
    return feature


def drop_columns(data, columns_to_drop):
    """Drop specified columns from the dataset."""
    data_dropped = data.drop(columns=columns_to_drop, errors='ignore')
    print(f"Columns dropped: {columns_to_drop}")
    return data_dropped


def plot_correlation_matrix(data, plots_folder, filename="correlation_matrix.png"):
    """Plot and save the correlation matrix of numeric columns in a dataset."""
    os.makedirs(plots_folder, exist_ok=True)

    numeric_data = data.select_dtypes(include=["number"])
    if numeric_data.empty:
        raise ValueError("The dataset contains no numeric columns for correlation analysis.")

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(plots_folder, filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Correlation matrix plot saved to {plot_path}")


def train_logistic_regression(data, target_column, excluded_columns, plots_folder):
    """Train a logistic regression model and save the confusion matrix with metrics."""
    os.makedirs(plots_folder, exist_ok=True)

    X = data.drop(columns=[target_column] + excluded_columns, errors='ignore')
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plot_path = os.path.join(plots_folder, "confusion_matrix_with_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

    return model


def plot_feature_importance(model, data, target_column, excluded_columns, plots_folder):
    """Plot and save feature importance for a logistic regression model."""
    os.makedirs(plots_folder, exist_ok=True)

    X = data.drop(columns=[target_column] + excluded_columns, errors='ignore')
    feature_names = X.columns

    importance = model.coef_[0]
    sorted_indices = importance.argsort()[::-1]
    sorted_features = feature_names[sorted_indices]
    sorted_importance = importance[sorted_indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()

    plot_path = os.path.join(plots_folder, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")