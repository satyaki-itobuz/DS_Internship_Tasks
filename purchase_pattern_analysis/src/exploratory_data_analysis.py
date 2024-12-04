# Import necessary libraries
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, pearsonr, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
from scipy import stats

logger= logging.getLogger(__name__)


def describe_data(dataframe: pd.DataFrame):
    try:
        """Generate descriptive statistics of the dataset."""
        return dataframe.describe()
    except Exception as description_error:
        return repr(description_error)


def display_skewness(dataframe: pd.DataFrame, title="Skewness of Dataset"):
    try:
        """Display skewness of numerical columns."""
        skewness = dataframe.select_dtypes(include=np.number).skew()
        skewness_df = pd.DataFrame(skewness, columns=['Skewness'])
        skewness_df = skewness_df.reset_index().rename(columns={'index': 'Feature'})
        skewness_df = skewness_df.sort_values(by='Skewness', ascending=False)
        return skewness_df
    except Exception as skewness_display_error:
        return repr(skewness_display_error)


def display_kurtosis(dataframe: pd.DataFrame, title="Kurtosis of Dataset"):
    try:
        """Display kurtosis of numerical columns."""
        kurtosis = dataframe.select_dtypes(include=np.number).kurtosis()
        kurtosis_df = pd.DataFrame(kurtosis, columns=['Kurtosis'])
        kurtosis_df = kurtosis_df.reset_index().rename(columns={'index': 'Feature'})
        kurtosis_df = kurtosis_df.sort_values(by='Kurtosis', ascending=False)
        return kurtosis_df
    except Exception as kurtosis_display_error:
        return repr(kurtosis_display_error)


def plot_countplot(column_name: str, dataframe: pd.DataFrame, color='mediumseagreen', title=None, xlabel=None, ylabel='Count', rotation=90, save_path=None):
    try:
        """Generate a count plot for a specified column."""
        plt.figure(figsize=(12, 6))
        sns.countplot(x=column_name, data=dataframe, color=color)
        plt.title(title if title else f'Count Plot of {column_name}')
        plt.xlabel(xlabel if xlabel else column_name)
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotation)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        #plt.show()
    except Exception as countplot_error:
        return repr(countplot_error)
    

def plot_boxplots(dataframe: pd.DataFrame, color='mediumseagreen', figsize=(20, 10), save_path=None):
    try:
        """Generate boxplots for numerical columns."""
        numerical_columns = dataframe.select_dtypes(include=np.number).columns
        plt.figure(figsize=figsize)
        
        for idx, column in enumerate(numerical_columns):
            plt.subplot(2, 3, idx + 1)
            sns.boxplot(y=dataframe[column], color=color)
            plt.title(f'Boxplot of {column}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        #plt.show()
    except Exception as boxplot_error:
        return repr(boxplot_error)

def plot_bar(x: str, y: str, dataframe: pd.DataFrame, xlabel=None, ylabel=None, color='mediumseagreen', title=None):
    try:
        """Generate a bar plot between two variables."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=x, y=y, data=dataframe, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()

    except Exception as barplot_error:
        return repr(barplot_error)


def plot_correlation_heatmap(dataframe: pd.DataFrame, figsize=(10, 8), annot=True, cmap="Greens", title="Correlation Heatmap"):
    try:
        """Generate a heatmap of correlations between numerical columns."""
        plt.figure(figsize=figsize)
        sns.heatmap(dataframe.corr(), annot=annot, cmap=cmap, fmt='.2f', linewidths=1, linecolor='black')
        plt.title(title)
        #plt.show()
    except Exception as heatmap_error:
        return repr(heatmap_error)


def eda_workflow(orders_data: pd.DataFrame, order_products_data: pd.DataFrame, aisles_data: pd.DataFrame, products_data: pd.DataFrame, merged_data: pd.DataFrame):
    # Descriptive statistics for each dataset
    #print("Descriptive Statistics of Orders Data:\n")
    #print(describe_data(orders_data))
    logger.info('Descriptive Statistics of Orders Data has been loaded')
    #print("Descriptive Statistics of Order Products Data:\n")
    #print(describe_data(order_products_data))
    logger.info('Descriptive Statistics of Order Products Data has been loaded')
    #print("Descriptive Statistics of Aisles Data:\n")
    #print(describe_data(aisles_data))
    logger.info('Descriptive Statistics of Aisles Data has been loaded')
    #print("Descriptive Statistics of Products Data:\n")
    #print(describe_data(products_data))
    logger.info('Descriptive Statistics of Products Data has been loaded')
    #print("Descriptive Statistics of Merged Data:\n")
    #print(describe_data(merged_data))
    logger.info('Descriptive Statistics of Merged Data has been loaded')
    
    # Skewness for each dataset
    #print("\nSkewness of Orders Data:\n", display_skewness(orders_data))
    #print("\nSkewness of Products Data:\n", display_skewness(products_data))
    #print("\nSkewness of Order Products Data:\n", display_skewness(order_products_data))
    #print("\nSkewness of Aisles Data:\n", display_skewness(aisles_data))
    #print("\nSkewness of Merged Data:\n", display_skewness(merged_data))
    logger.info('Skewness of all datasets has beed loaded.')

    # Kurtosis for each dataset
    #print("\nKurtosis of Products Data:\n", display_kurtosis(products_data))
    #print("\nKurtosis of Order Products Data:\n", display_kurtosis(order_products_data))
    #print("\nKurtosis of Orders Data:\n", display_kurtosis(orders_data))
    #print("\nKurtosis of Aisles Data:\n", display_kurtosis(aisles_data))
    #print("\nKurtosis of Merged Data:\n", display_kurtosis(merged_data))
    logger.info('Kurtosis of all datasets has been loaded.')

    # Count plots for relevant columns
    plot_countplot('department_id', products_data, color='mediumseagreen', title='Department ID Distribution', xlabel='Department ID')
    plot_countplot('aisle_id', products_data, color='mediumseagreen', title='Aisle ID Distribution', xlabel='Aisle ID')
    plot_countplot('order_number', orders_data, color='mediumseagreen', title='Order Number Distribution', xlabel='Order Number')

    # Box plots for each dataset
    plot_boxplots(merged_data, color='mediumseagreen', save_path='plots/BoxPlots_Merged_Data.png')
    plot_boxplots(order_products_data, color='mediumseagreen', save_path='plots/BoxPlots_Order_Products_Data.png')
    plot_boxplots(products_data, color='mediumseagreen', save_path='plots/BoxPlots_Products_Data.png')
    plot_boxplots(orders_data, color='mediumseagreen', save_path='plots/BoxPlots_Orders_Data.png')
    plot_boxplots(aisles_data, color='mediumseagreen', save_path='plots/BoxPlots_Aisles_Data.png')

    # Bar plots for relationships between variables
    plot_bar(x='order_number', y='order_hour_of_day', dataframe=orders_data, xlabel='Order Number', ylabel='Order Hour of Day', title='Order Number vs Order Hour of Day')
    plot_bar(x='order_number', y='order_dow', dataframe=orders_data, xlabel='Order Number', ylabel='Order Dow', title='Order Number vs Days of Week')

    # Correlation heatmaps
    plot_correlation_heatmap(products_data, figsize=(12, 10))
    plot_correlation_heatmap(order_products_data, figsize=(12, 10))
    plot_correlation_heatmap(orders_data, figsize=(12, 10))
    plot_correlation_heatmap(merged_data, figsize=(12, 10))
