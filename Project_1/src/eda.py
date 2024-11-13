import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
from scipy.stats import f_oneway, kruskal, chi2_contingency
import scipy.stats


def stratified_sample(df:pd.DataFrame, stratify_col:str, frac:float) -> pd.DataFrame:
    """
    Applying Stratified sampling on the dataset
    """
    stratified_df = df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
    stratified_df = stratified_df.reset_index(drop=True)
    return stratified_df

def plot_histogram(df:pd.DataFrame,col:str):
    """
    Histogram plot of a column in the dataset
    """
    df[col].hist(bins=20)
    plt.title(f"histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('count')
    plt.show()

def plot_boxplot(df:pd.DataFrame,col:str):
    """
    Boxplot of a column in the dataset
    """
    sns.boxplot(data=df,x=col)
    plt.title(f"boxplot of {col}")
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.show()

def skew_kurtosis(df:pd.DataFrame,col:str) -> list:
    """
    Finding Skewness and kurtosis of the dataset
    """
    res=[]
    skew=df[col].skew()
    kurtosis=df[col].kurtosis()
    res.append(skew,kurtosis)

    return res

def pie_chart(df:pd.DataFrame,col:str):
    """
    Pie chart of column in the dataset 
    """
    df_counts=df[col].value_counts()
    plt.pie(df_counts,labels=df_counts.index,autopct="%1.1f%")
    plt.title(f'{col} distribution (%)')
    plt.axis('equal')
    plt.show()

def bar_plot(df:pd.DataFrame,col:str):
    """
    Bar chart of column in the dataset 
    """
    df_counts=df[col].value_counts()
    df_plot=df_counts.plot(kind='bar')
    plt.xlabel(f'{col}')
    plt.ylabel('count')
    plt.xticks(rotation=45)
    plt.show()

def box_plot_bivariate(df:pd.DataFrame,col1:str,col2:str):
    """
    Box plot for bivariate analysis
    """
    sns.boxplot(data=df,x=col1,y=col2)
    plt.xticks(rotation=45)
    plt.show()

def cross_tab_heatmap(df:pd.DataFrame,col1:str,col2:str):
    """
    Cross tab and heatmap for bivariate analysis
    """
    new_data=pd.crosstab(df[col1],df[col2])
    sns.heatmap(new_data,cmap='viridis',annot=True,fmt='d')
    plt.title(f'{col1} vs {col2}')
    plt.xlabel(f'{col1}')
    plt.ylabel(f'{col2}')
    plt.show()

def countplot(df:pd.DataFrame,col:str):
    """
    Countplot of a column
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

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

def apply_binning(stratified_df: pd.DataFrame, col1:str, col2:str):
    """
    Applying binning and bucketing by calling the other functions
    """

    required_columns = ['order_hour_of_day', 'order_number', 'days_since_prior_order', 'order_dow']
    for col in required_columns:
        if col not in stratified_df.columns:
            raise ValueError(f"Missing column: {col}")


    stratified_df['time_of_day'] = stratified_df['order_hour_of_day'].apply(assign_time_period)
    stratified_df['customer_type'] = stratified_df['order_number'].apply(order_frequency)
    stratified_df['order_recency'] = stratified_df['days_since_prior_order'].apply(days_since_order_category)
    stratified_df['peak_category'] = stratified_df.apply(peak_hours, axis=1)




def analyze_reorder_probability(df: pd.DataFrame, feature1, feature2, threshold=100):
    """
    Finding reorder probability and heatmap together 
    """

    pivot = df.groupby([feature1, feature2])['reordered'].agg(['mean', 'count']).reset_index()
    pivot = pivot[pivot['count'] > threshold]  
    
    pivot_table = pivot.pivot(index=feature1, columns=feature2, values='mean')
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title(f'Reorder Probability: {feature1} vs {feature2}')
    plt.show()
    
    return pivot_table


def print_test_results(test_name, statistic, p_value, significance_level=0.05):
    """
    Print the test results with the statistic, p-value, and significance test.
    """
    print(f"\n{test_name}:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at {significance_level} level: {p_value < significance_level}")

def run_statistical_tests(stratified_df: pd.DataFrame, col1:str, col2:str, significance_level=0.05):
    """
    Perform and print the results of ANOVA, Kruskal-Wallis, and Chi-Square tests.
    """

    if col1 not in stratified_df.columns or col2 not in stratified_df.columns:
        raise ValueError(f"Columns {col1} or {col2} are missing from the DataFrame.")
    
    stratified_df_clean = stratified_df.dropna(subset=[col1, col2])

    departments = stratified_df_clean['department'].unique()
    days_by_dept = [stratified_df_clean[stratified_df_clean['department'] == dept][col2] 
                    for dept in departments]
    
    f_stat, p_val = f_oneway(*days_by_dept)
    print_test_results("ANOVA Test: Days Since Prior Order across Departments", f_stat, p_val, significance_level)

    h_stat, p_val = scipy.stats.kruskal(*days_by_dept)
    print_test_results("Kruskal-Wallis H-test: Days Since Prior Order across Departments", h_stat, p_val, significance_level)

    dept_reorder_counts = pd.crosstab(stratified_df_clean['department'], stratified_df_clean['reordered'])
    
    chi2, p_value, dof, expected = chi2_contingency(dept_reorder_counts)
    print("\nChi-square test results for Department vs Reordered:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected frequencies:\n{expected}")
    print(f"Significant at {significance_level} level: {p_value < significance_level}")

# Can add save = True in function as a flag if user wants to save the plot or just show the plot