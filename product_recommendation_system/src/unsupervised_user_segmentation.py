import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def get_cluster_numbers_for_kmeans(scaled_features):
    scores = []
    silhouette_scores = []
    n_components_range = range(2, 11)

    for n_components in n_components_range:
        
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        kmeans.fit(scaled_features)
        
        scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, scores, marker='o', label="(Elbow Method")
    plt.show()
    plt.plot(n_components_range, silhouette_scores, marker='o', color='r', label="Silhouette Score")
    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.title("Elbow Method and Silhouette Score for Kmeans")
    plt.show()


def user_segmentation_using_clustering_with_kmeans(features, scaled_features,user_profiles):
    
    # check elbow method and sillhoutte score
    get_cluster_numbers_for_kmeans(scaled_features)

    # clustering using kmeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(scaled_features)
    features['cluster'] = kmeans.labels_
    user_profiles['cluster']  = kmeans.labels_

    feature_set = [
    'total_orders',
    'avg_days_between_orders',
    'unique_products',
    'basket_size',
    'repeat_purchase_percentage',
    'avg_purchase_hour',
    'weekend_shopper_percentage',
    'is_weekend']

    cluster_means = user_profiles.groupby('cluster')[feature_set].mean()

    normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    labels = feature_set
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 


    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for cluster in normalized_means.index:
        values = normalized_means.loc[cluster].tolist()
        values += values[:1] 
        ax.plot(angles, values, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_yticklabels(['0.2', '0.5', '0.8'], color="grey", size=8)
    ax.set_ylim(0, 1)


    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Radar Plot of Cluster Means', size=14, y=1.1)
    plt.show()


def get_cluster_numbers_for_gmm(scaled_features):  
    n_components_range = range(2, 10)
    bics = []
    silhouette_scores = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(scaled_features)
        bics.append(gmm.bic(scaled_features))

        labels = gmm.predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        silhouette_scores.append(score)

        fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("BIC", color="tab:blue")
    ax1.plot(n_components_range, bics, marker='o', label='BIC', color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    ax2.plot(n_components_range, silhouette_scores, marker='o', label='Silhouette Score', color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    plt.title("BIC and Silhouette Scores for Different Number of Components")
    plt.show()


def user_segmentation_using_clustering_with_GMM(features, scaled_features,user_profiles):
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(scaled_features)

    user_profiles['gmm_cluster'] = gmm.predict(scaled_features)
    # Assuming 'gmm' is your fitted GMM model and 'df' is your DataFrame with data points
    probabilities = gmm.predict_proba(scaled_features)

    # Create empty lists to store the top 3 clusters and probabilities for each point
    top_2_clusters = []
    top_2_probabilities = []

    for prob in probabilities:
        # Get the indices of the top 3 clusters and their corresponding probabilities
        top_indices = np.argsort(prob)[::-1][:2]  # Sort probabilities in descending order and get top 3
        top_probs = prob[top_indices]
        
        # Round the probabilities to 2 decimal places
        top_probs_rounded = np.round(top_probs * 100, 2)  # Convert to percentages and round
        
        # Store the results for the current data point
        top_2_clusters.append(top_indices)
        top_2_probabilities.append(top_probs_rounded)

    # Add the top 3 clusters and their probabilities to the DataFrame
    user_profiles[['Top Cluster 1', 'Top Cluster 2']] = pd.DataFrame(top_2_clusters, index=user_profiles.index)
    user_profiles[['Top Cluster 1 Probability (%)', 'Top Cluster 2 Probability (%)']] = pd.DataFrame(top_2_probabilities, index=user_profiles.index)