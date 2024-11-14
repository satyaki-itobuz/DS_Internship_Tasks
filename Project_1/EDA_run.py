from src.data_preprocessing import *
from src.eda import *

# storing the location of the csv files 
data1= "/Users/it012305/Downloads/project_1_dataset/aisles.csv"
data2= "/Users/it012305/Downloads/project_1_dataset/orders.csv"
data3= "/Users/it012305/Downloads/project_1_dataset/order_products.csv"
data4= "/Users/it012305/Downloads/project_1_dataset/products.csv"
data5= "/Users/it012305/Downloads/project_1_dataset/departments.csv"

# performing preprocessing steps

# loading data
aisles=load_data(data1)
orders=load_data(data2)
order_pro=load_data(data3)
pro=load_data(data4)
dept=load_data(data5)

# merging data
pro1=merging_data(df1=pro,df2=dept,on_column='department_id')
pro2=merging_data(df1=pro1,df2=aisles,on_column='aisle_id')
pro3=merging_data(df1=pro2,df2=order_pro,on_column='product_id')
df=merging_data(df1=pro3,df2=orders,on_column='order_id')

# cleaning data
df=clean_data(df=df,col='eval_set')

# missing value treatment
df=missing_values(df=df,col='days_since_prior_order')

# stratified sampling 
df=stratified_sample(df=df,stratify_col='product_name',frac=0.1)

# histogram of numerical columns 
columns_to_plot = ['add_to_cart_order','order_number','days_since_prior_order','order_hour_of_day'] 
for i in range(len(columns_to_plot)):
    plot_histogram(df=df,col=columns_to_plot[i],save=True)


# box plot of numerical columns 
cols_to_plot = ['order_hour_of_day','days_since_prior_order'] 
for i in range(len(cols_to_plot)):
    plot_boxplot(df=df,col=cols_to_plot[i],save=True)

# skewness and kurtosis of the numerical columns 
cols_to_plot_2 = ['add_to_cart_order', 'order_number', 'order_hour_of_day', 'days_since_prior_order', 'order_dow']
for col in cols_to_plot_2:
    result = skew_kurtosis(df, col)
    print(f"Skewness and Kurtosis for '{col}': {result}")


# pie chart of the columns 
pie_columns=['department','aisle','order_dow']
for col in cols_to_plot_2:
    pie_chart(df=df,col=col)

# bar plot of categorical columns 
bar_columns=['department','aisle','order_dow']
for col in bar_columns:
    bar_plot(df=df,col=col)

# box plot for bivariate analysis
box_plot_bivariate(df=df,col1='department',col2='days_since_prior_order')
box_plot_bivariate(df=df,col1='aisle',col2='add_to_cart_order')

# contigency table with heatmap 
cross_tab_heatmap(df=df,col1='order_hour_of_day',col2='order_dow',save=True)
cross_tab_heatmap(df=df,col1='department',col2='reordered',save=True)
cross_tab_heatmap(df=df,col1='order_hour_of_day',col2='department',save=True)

# reorder rate by category heatmap 
plot_reorder_rate_by_category(df=df)

# mean of add to cart otder by hour heatmap
plot_mean_add_to_cart_order_by_hour(df=df) #asdad

# reorder rate heatmap
plot_reorder_rate_heatmap(df=df)

# reorder probability heatmap
analyze_reorder_probability(df=df,feature1='order_dow',feature2='order_hour_of_day')

# binning and bucketing
stratified_df=apply_binning(stratified_df=df)

print(stratified_df[['time_of_day', 'customer_type', 'order_recency', 'peak_category']].head())

# countplot of binning and bucketing
countplot(df=df,col=stratified_df['time_of_day'])
countplot(df=df,col=stratified_df['customer_type'])
countplot(df=df,col=stratified_df['order_recency'])
countplot(df=df,col=stratified_df['peak_category'])


numerical_cols = ['add_to_cart_order', 'reordered', 'order_number', 
                  'order_dow', 'order_hour_of_day', 'days_since_prior_order']

# correlation matrix of numerical columns 
plot_correlation_matrix(df, numerical_cols)

# statistical tests 
run_statistical_tests(stratified_df=df,col1='department',col2='days_since_prior_order')









