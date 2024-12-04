import os
from src.feature_engineering import (
    load_datasets,
    save_csv,
    average_days_between_purchases,
    product_purchase_frequency,
    purchase_stats,
    product_reorder_rate,
    users_general_reorder_rate,
    avg_add_to_cart_order,
    drop_columns,
    plot_correlation_matrix,
    train_logistic_regression,
    plot_feature_importance,
)

if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.abspath(os.path.join(script_dir, "data"))
    results_folder = os.path.abspath(os.path.join(script_dir, "results"))
    processed_data_path = os.path.join(data_folder, "processed_data.csv")
    output_file = os.path.join(data_folder, "feature_engineered_data.csv")

    # Load data
    sampled_data = load_datasets(processed_data_path)

    # Generate features
    features = [
        average_days_between_purchases(sampled_data, results_folder),
        product_purchase_frequency(sampled_data, results_folder),
        purchase_stats(sampled_data, results_folder),
        product_reorder_rate(sampled_data, results_folder),
        users_general_reorder_rate(sampled_data, results_folder),
        avg_add_to_cart_order(sampled_data, results_folder),
    ]

    # Merge all features with the dataset
    for feature in features:
        if 'user_id' in feature.columns and 'product_id' in feature.columns:
            merge_keys = ['user_id', 'product_id']
        elif 'user_id' in feature.columns:
            merge_keys = ['user_id']
        elif 'product_id' in feature.columns:
            merge_keys = ['product_id']
        else:
            raise ValueError(f"Unexpected feature structure: {feature.columns.tolist()}")

        sampled_data = sampled_data.merge(feature, on=merge_keys, how='left')

    # Columns to drop
    columns_to_drop = [
        'add_to_cart_order_x',
        'eval_set',
        'days_since_prior_order',
        'product_name',
        'aisle',
        'department',
        'aisle_id',
        'department_id',
        'order_hour_of_day',
        'order_dow',
        'add_to_cart_order_y'
    ]

    # Drop columns
    sampled_data = drop_columns(sampled_data, columns_to_drop)

    # Save the feature-engineered dataset
    save_csv(sampled_data, data_folder, "feature_engineered_data.csv")
    print(f"Feature-engineered dataset saved to {output_file}")

    # Plot correlation matrix of the updated dataset
    plots_folder = os.path.abspath(os.path.join(script_dir, "plots"))
    plot_correlation_matrix(sampled_data, plots_folder)
    print("Correlational Matrix Plot is Saved")

    # Target column
    target_column = "reordered"

    # Exclude 'interval_std_dev' from features
    excluded_columns = ['interval_std_dev', 'days_since_prior_order_binned']

    # Train model and save confusion matrix
    model = train_logistic_regression(
        sampled_data,
        target_column=target_column,
        excluded_columns=excluded_columns,
        plots_folder=plots_folder
    )
    print("Model Trained")

    # Plot and save feature importance
    plot_feature_importance(
        model,
        sampled_data,
        target_column=target_column,
        excluded_columns=excluded_columns,
        plots_folder=plots_folder
    )
    print("Feature Importance Plot is Saved")