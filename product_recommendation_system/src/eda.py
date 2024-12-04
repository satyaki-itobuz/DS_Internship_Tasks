import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
import json
from config import STRATIFICATION_FRACTION
from config import EDA_PLOTS_LOCATION 
from config import RANDOM_STATE
import logging

logger=logging.getLogger(__name__)

def stratified_sample(df:pd.DataFrame, stratify_col:str, frac:float) -> pd.DataFrame:
    """
    Applying Stratified sampling on the dataset
    """
    try:
        stratified_df = df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=STRATIFICATION_FRACTION, random_state=RANDOM_STATE))
        stratified_df = stratified_df.reset_index(drop=True)
        return stratified_df
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def plot_histogram(df:pd.DataFrame,col:str,save:bool):
    """
    Histogram plot of a column in the dataset
    """
    try:
        df[col].hist(bins=20)
        plt.title(f"histogram of {col}")
        plt.xlabel(col)
        plt.ylabel('count')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}hist_{col}.png')
    #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def plot_boxplot(df:pd.DataFrame,col:str,save:bool):
    """
    Boxplot of a column in the dataset
    """
    try:
        sns.boxplot(data=df,x=col)
        plt.title(f"boxplot of {col}")
        plt.xlabel(col)
        plt.ylabel('frequency')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}box_{col}.png')
    #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def skew_kurtosis(df:pd.DataFrame,col:str) -> list:
    """
    Finding Skewness and kurtosis of the dataset
    """
    try:
        skew=df[col].skew()
        kurtosis=df[col].kurtosis()
        res=[skew,kurtosis]

        return res
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def pie_chart(df:pd.DataFrame,col:str='reordered',save:bool=True):
    """
    Pie chart of column in the dataset 
    """
    try:
        plt.figure(figsize=(12,8))
        df_counts=df[col].value_counts()
        plt.pie(df_counts,labels=df_counts.index,autopct="%1.1f%%")
        plt.title(f'{col} distribution (%)')
        plt.axis('equal')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}pie_{col}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def bar_plot(df:pd.DataFrame,col:str,save:bool=True):
    """
    Bar chart of column in the dataset 
    """
    try:
        df_counts=df[col].value_counts()
        df_plot=df_counts.plot(kind='bar')
        plt.xlabel(f'{col}')
        plt.ylabel('count')
        plt.xticks(rotation=45)
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}bar_{col}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def box_plot_bivariate(df:pd.DataFrame,col1:str,col2:str,save:bool=True):
    """
    Box plot for bivariate analysis
    """
    try:
        plt.figure(figsize=(10,20))
        sns.boxplot(data=df,x=col1,y=col2)
        plt.xticks(rotation=45)
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}box_bivariate_{col1}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def cross_tab_heatmap(df:pd.DataFrame,col1:str,col2:str,save:bool):
    """
    Cross tab and heatmap for bivariate analysis
    """
    try:
        plt.figure(figsize=(15, 8))
        new_data=pd.crosstab(df[col1],df[col2])
        sns.heatmap(new_data,cmap='viridis',annot=True,fmt='d')
        plt.title(f'{col1} vs {col2}')
        plt.xlabel(f'{col2}')
        plt.ylabel(f'{col1}')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}cross_heatmap{col1}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def countplot(df:pd.DataFrame,col:str,save:bool=True):
    """
    Countplot of a column
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}count_plot_for_binning.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)

def assign_time_period(hour):
    """
    Classifying time of day based on hour of the day 
    """
    if 5 <= hour <= 11:
        return 'Morning'
    elif hour < 0 or hour > 23:
        return 'Invalid hour'
    elif 12 <= hour <= 16:
        return 'Afternoon'
    elif 17 <= hour <= 21:
        return 'Evening'
    else:
        return 'Night'


def order_frequency(order_num):
    """
    Order frequency based on order number
    """
    if order_num == 1:
        return 'First Time'
    elif order_num < 1:
        return 'Invalud order number'
    elif 2 <= order_num <= 5:
        return 'New Customer'
    elif 6 <= order_num <= 15:
        return 'Regular Customer'
    else:
        return 'Loyal Customer'

def days_since_order_category(days):
    """
    Reorder time period 
    """
    if pd.isna(days):
        return 'First Order'
    elif days < 0:
        return 'Invalid days'
    elif days <= 7:
        return 'Within Week'
    elif days <= 14:
        return 'Within Fortnight'
    elif days <= 30:
        return 'Within Month'
    else:
        return 'More than Month'


def peak_hours(row):
    """
    Peak hours time period
    """
    hour = row['order_hour_of_day']
    day = row['order_dow']
    
    if hour < 0 or hour > 23 or day < 0 or day > 6:
        return 'Invalid Time/Day'
    
    if day in [5, 6]: 
        if 10 <= hour <= 18:
            return 'Weekend Peak'
        else:
            return 'Weekend Off-Peak'
    else:
        if (8 <= hour <= 10) or (17 <= hour <= 19):
            return 'Weekday Peak'
        else:
            return 'Weekday Off-Peak'

def apply_binning(stratified_df: pd.DataFrame):
    """
    Applies binning and bucketing by calling the other functions.
    Adds new columns to the DataFrame based on time of day, customer type, etc.
    """
    required_columns = ['order_hour_of_day', 'order_number', 'days_since_prior_order', 'order_dow']
    
    for col in required_columns:
        if col not in stratified_df.columns:
            raise ValueError(f"Missing column: {col}")
    
    stratified_df['time_of_day'] = stratified_df['order_hour_of_day'].apply(assign_time_period)
    stratified_df['customer_type'] = stratified_df['order_number'].apply(order_frequency)
    stratified_df['order_recency'] = stratified_df['days_since_prior_order'].apply(days_since_order_category)
    stratified_df['peak_category'] = stratified_df.apply(peak_hours, axis=1)
    
    return stratified_df

def analyze_reorder_probability(df: pd.DataFrame, feature1, feature2, threshold=100,save:bool=True):
    """
    Finding reorder probability and heatmap together 
    """
    try:
        plt.figure(figsize=(12,10))
        pivot = df.groupby([feature1, feature2])['reordered'].agg(['mean', 'count']).reset_index()
        pivot = pivot[pivot['count'] > threshold]  
        
        pivot_table = pivot.pivot(index=feature1, columns=feature2, values='mean')
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title(f'Reorder Probability: {feature1} vs {feature2}')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}analyze_{feature1}.png')
        #plt.show()
        
        return pivot_table
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)


def print_test_results(test_name, statistic, p_value, significance_level=0.05):
    """
    Print the test results with the statistic, p-value, and significance test.
    """
    print(f"\n{test_name}:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at {significance_level} level: {p_value < significance_level}")


def run_statistical_tests(stratified_df: pd.DataFrame, col1: str, col2: str, significance_level=0.05, json_file_path=None):
    """
    Perform and print the results of ANOVA, Kruskal-Wallis, and Chi-Square tests.
    Optionally saves the results to a JSON file.
    """

    def convert_to_python_types(obj):
        """Recursively converts NumPy types to native Python types (e.g., np.bool_ to bool)."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj
    
    test = {}

    if col1 not in stratified_df.columns or col2 not in stratified_df.columns:
        raise ValueError(f"Columns {col1} or {col2} are missing from the DataFrame.")
    
    stratified_df_clean = stratified_df.dropna(subset=[col1, col2])

    departments = stratified_df_clean['department'].unique()
    days_by_dept = [stratified_df_clean[stratified_df_clean['department'] == dept][col2] 
                    for dept in departments]

    
    f_stat, p_val = f_oneway(*days_by_dept)
    test["ANOVA"] = {
        "test_name": "ANOVA Test: Days Since Prior Order across Departments",
        "statistic": f_stat,
        "p_value": p_val,
        "significant": p_val < significance_level
    }
    
    
    h_stat, p_val = kruskal(*days_by_dept)
    test["KRUSKAL"] = {
        "test_name": "Kruskal-Wallis Test: Days Since Prior Order across Departments",
        "statistic": h_stat,
        "p_value": p_val,
        "significant": p_val < significance_level
    }

    
    dept_reorder_counts = pd.crosstab(stratified_df_clean['department'], stratified_df_clean['reordered'])
    chi2, p_value, dof, expected = chi2_contingency(dept_reorder_counts)
    test["Chi-Square"] = {
        "test_name": "Chi-square Test: Department vs Reordered",
        "statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "expected_frequencies": expected.tolist(), 
        "significant": p_value < significance_level
    }

  
    for test_name, result in test.items():
        print_test_results(result["test_name"], result["statistic"], result["p_value"], significance_level)

    test = convert_to_python_types(test)
    
    if json_file_path:
        with open(json_file_path, 'w') as f:
            json.dump(test, f, indent=4)

    return test



def plot_reorder_rate_heatmap(df: pd.DataFrame, hour_col: str = 'order_hour_of_day', dow_col: str = 'order_dow', reorder_col: str = 'reordered', figsize: tuple = (15, 10),save:bool=True):
    """
    Creates a heatmap for reorder rates by hour and day of the week
    """
    try:
        pivot_table = df.pivot_table(
            values=reorder_col,
            index=hour_col,
            columns=dow_col,
            aggfunc='mean'
        )
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Reorder Rate by Hour and Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}rate_heatmap_{hour_col}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)


def plot_mean_add_to_cart_order_by_hour(df: pd.DataFrame, hour_col: str = 'order_hour_of_day', cart_order_col: str = 'add_to_cart_order', figsize: tuple = (15, 6),save:bool=True):
    """
    Plots the mean add_to_cart_order by hour of the day
    """
    try:
        mean_cart_order = df.groupby(hour_col)[cart_order_col].mean()

        plt.figure(figsize=figsize)
        mean_cart_order.plot(kind='line', marker='o')
        plt.title('Average Add to Cart Order by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Add to Cart Order')
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}mean_add_{hour_col}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)


def plot_reorder_rate_by_category(df: pd.DataFrame, category_col: str = 'department', target_col: str = 'reordered', colors: list = ['#1f77b4'], figsize: tuple = (15, 6),save:bool=True):
    """
    Creates a bar plot showing the reorder rate by category (e.g., department)
    """
    try:
        reorder_rate = pd.crosstab(df[category_col], df[target_col], normalize='index') * 100
        
        plt.figure(figsize=figsize)
        reorder_rate[1].sort_values(ascending=False).plot(kind='bar', color=colors[0])
        plt.title(f'Reorder Rate by {category_col.capitalize()}')
        plt.xlabel(category_col.capitalize())
        plt.ylabel('Reorder Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}reorder_rate_{target_col}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)


def plot_correlation_matrix(df: pd.DataFrame, numerical_cols: list, figsize: tuple = (10, 8), cmap: str = 'coolwarm',save:bool=True):
    """
    Plots a heatmap for the correlation matrix of numerical variables
    """
    try:
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        if save:
            plt.savefig(f'{EDA_PLOTS_LOCATION}correlation_matrix{numerical_cols}.png')
        #plt.show()
    except Exception as FileNotFoundError:
        logging.error("An error occured")
        return repr(FileNotFoundError)