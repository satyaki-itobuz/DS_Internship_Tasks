
import pandas as pd
from src.unsupervised_data_preprocessing import preprocess_unsupervised_data, stratified_sample, aggregate_user_data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.unsupervised_user_segmentation import user_segmentation_using_clustering_with_GMM, user_segmentation_using_clustering_with_kmeans

def main():
    # Load and preprocess data
    df_products = pd.read_csv('../../eda/datasets/project_1_dataset/products.csv')
    df_ordered_products = pd.read_csv('../../eda/datasets/project_1_dataset/order_products.csv')
    df_orders = pd.read_csv('../../eda/datasets/project_1_dataset/orders.csv')

    # merge data
    df = pd.merge(df_orders, df_ordered_products, on='order_id', how='inner')
    df.drop(columns=['eval_set', 'order_number', 'add_to_cart_order','reordered'], inplace=True, axis=1)
    df = pd.merge(df, df_products, on='product_id', how='inner')

    # preprocess data
    df = preprocess_unsupervised_data(df)

    # stratify data
    stratified_df = stratified_sample(df, 'user_id', frac=0.3)

    # Creating unique user profiles
    user_profiles = aggregate_user_data(stratified_df)

    # scale features
    features = user_profiles.drop(columns=['user_id'], axis=1)
    features.dropna(inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # user segmentation
    user_segmentation_using_clustering_with_kmeans(features, scaled_features,user_profiles)
    user_segmentation_using_clustering_with_GMM(features, scaled_features,user_profiles)


    # final results : eda on clustering


if __name__ == "__main__":
    main()