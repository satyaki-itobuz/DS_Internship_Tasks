# DEPENDENCIES
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


#GET LOGGER 
logger = logging.getLogger(__name__)


def customer_segmentation_analysis(merged_data:pd.DataFrame, aisle_data:pd.DataFrame) -> dict :
    """
        Description

        Arguments:
        ----------
            
            aisle_data {DataFrame} :


        Errors:
        -------
            TypeError :

            InsufficientData : 

        Return:
        -------
            { dict } : 
    """
    # Type Checking 
    if not isinstance(merged_data, pd.DataFrame):
        logger.warning((f"Expected a pandas DataFrame object, got : {type(merged_data)} instead"))
        return repr(TypeError(f"Expected a pandas DataFrame object, got : {type(merged_data)} instead"))

    if not isinstance(aisle_data, pd.DataFrame):
        logger.warning((f"Expected a pandas DataFrame object, got : {type(aisle_data)} instead"))
        return repr(TypeError(f"Expected a pandas DataFrame object, got : {type(aisle_data)} instead"))

    # Data Validation
    if (len(merged_data) < 2):
        logger.warning(f"InsufficientData : As the input products_data if of length : {len(merged_data)}, no further processing possible")
        return repr(f"InsufficientData : As the input products_data if of length : {len(merged_data)}, no further processing possible")

    if (len(aisle_data) < 2):
        logger.warning(f"InsufficientData : As the input products_data if of length : {len(aisle_data)}, no further processing possible")
        return repr(f"InsufficientData : As the input products_data if of length : {len(aisle_data)}, no further processing possible")

    try:
        train_df, test_df = train_test_split(merged_data, test_size=0.99, stratify=merged_data['user_id'], random_state=42)

        # Display the shape of the training and testing sets
        logger.info(f"Training set size: {train_df.shape[0]} rows")
        # print(f"Training set size: {train_df.shape[0]} rows")
        logger.info(f"Testing set size: {test_df.shape[0]} rows")
        # print(f"Testing set size: {test_df.shape[0]} rows")

        # Optional: Check the distribution of user_id in both sets
        logger.info("User ID distribution in training set")
        # print("User ID distribution in training set:")
        logger.info(train_df['user_id'].value_counts(normalize=True))
        # print(train_df['user_id'].value_counts(normalize=True))


        #logger.info("User ID distribution in training set:")
        # print("User ID distribution in testing set:")
        # logger.info(test_df['user_id'].value_counts(normalize=True))
        # print(test_df['user_id'].value_counts(normalize=True))
        # Convert the DataFrame to a dictionary with columns as keys
        df_dict = aisle_data.to_dict(orient='dict')['aisle']

        # Mapped the aisle names to the merged dataframe according to aisle_id
        merged_data['aisle'] = merged_data['aisle_id'].map(df_dict)
        
        # Cross-tabulation
        cross_df = pd.crosstab(merged_data.user_id, merged_data.aisle)
        
        #Performed row-wise normalization on the cross_df DataFrame, making each row sum up to 1.  
        df = cross_df.div(cross_df.sum(axis=1), axis=0)

        # Principal Component Analysis
        pca = PCA(n_components=10)
        df_pca = pca.fit_transform(df)
        df_pca = pd.DataFrame(df_pca)
        # df_pca.head()

        Sum_of_squared_distances = []
        K = range(1,10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df_pca)
            Sum_of_squared_distances.append(km.inertia_)
        
        # Plotting Elbow score
        plt.subplots(figsize = (8, 5))
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.savefig('./plots/elbow_plot.jpeg', bbox_inches = 'tight')
        
        # K-Means Clustering 
        clusterer = KMeans(n_clusters=5,random_state=42).fit(df_pca)
        centers = clusterer.cluster_centers_
        c_preds = clusterer.predict(df_pca)
        # logger.info(centers)
        # print(centers)

        # Taking PC-1 and PC-2 for further analysis
        temp_df = df_pca.iloc[:, 0:2]
        temp_df.columns = ["pc1", "pc2"]
        temp_df['cluster'] = c_preds
        temp_df.head()
        
        # Cluster Visualization
        fig, ax = plt.subplots(figsize = (8, 5))
        ax = sns.scatterplot(data = temp_df, x = "pc1", y = "pc2", hue = "cluster")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Cluster Visualization")
        plt.savefig('./plots/cluster_visualization.jpeg', bbox_inches = 'tight')

        # Cluster-Rankings
        cross_df['cluster'] = c_preds

        cluster1 = cross_df[cross_df.cluster == 0]
        cluster2 = cross_df[cross_df.cluster == 1]
        cluster3 = cross_df[cross_df.cluster == 2]
        cluster4 = cross_df[cross_df.cluster == 3]
        cluster5 = cross_df[cross_df.cluster == 4]

        #
        cluster1.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
        cluster2.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
        cluster3.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
        cluster4.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
        cluster5.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]

        cluster1.shape
        cluster2.shape
        cluster3.shape
        cluster4.shape
        cluster5.shape

        # 1.  Cluster1 has 55851 columns with very strong inclination towards fresh vegetables followed by fresh fruits.
        # 2.  Cluster2 has 37837 columns with very strong inclination towards fresh fruits followed by fresh vegetables.
        # 3.  Cluster3 has 99157 columns with moderately strong inclination towards fresh fruits followed by fresh vegetables.
        # 4.  Cluster4 has 7947 columns with very strong inclination towards packaged produce followed by fresh fruits.
        # 5.  Cluster5 has 5417 columns with strong inclination towards water seltzer sparkling water followed by fresh fruits. 
        # 
        return {'cluster-1' : cluster1,
                'cluster-2' : cluster2,
                'cluster-3' : cluster3,
                'cluster-4' : cluster4,
                'cluster-5' : cluster5}

    except Exception as CustomerSegmentationAnalysisError:
        logger.error(f"CustomerSegmentationAnalysisError: While performing customer segmentation analysis, got error: {repr(CustomerSegmentationAnalysisError)}")
        return (f"CustomerSegmentationAnalysisError: While performing customer segmentation analysis, got error: {repr(CustomerSegmentationAnalysisError)}")
        raise



