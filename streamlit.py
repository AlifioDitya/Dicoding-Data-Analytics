# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.simplefilter(action='ignore', category=FutureWarning)

# Set page configuration
st.set_page_config(page_title='E-Commerce Data Analysis Dashboard', layout='wide')

# Title and Introduction
st.title('E-Commerce Data Analysis: Brazilian E-Commerce Dataset by Olist')
st.markdown("""
**by E. Alifio Ditya**

In this project, we will tackle some business-related questions using data analysis skills applied to the Olist E-Commerce Dataset. The dataset contains information about orders, customers, products, and reviews from 2016 to 2018.

**Dataset source**: [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce)
""")

# Sidebar
st.sidebar.title("Enrique Alifio Ditya")
st.sidebar.info("Google Bangkit Cohort 2024")

DATA_PATH = './data/'

# Define file paths
CUSTOMERS = DATA_PATH + 'customers_dataset.csv'
GEOLOCATION = DATA_PATH + 'geolocation_dataset.csv'
ORDER_ITEMS = DATA_PATH + 'order_items_dataset.csv'
PAYMENTS = DATA_PATH + 'order_payments_dataset.csv'
ORDERS = DATA_PATH + 'orders_dataset.csv'
REVIEWS = DATA_PATH + 'order_reviews_dataset.csv'
TRANSLATION = DATA_PATH + 'product_category_name_translation.csv'
PRODUCTS = DATA_PATH + 'products_dataset.csv'
SELLERS = DATA_PATH + 'sellers_dataset.csv'

# Load data
@st.cache_data
def load_data():
    orders = pd.read_csv(ORDERS)
    customers = pd.read_csv(CUSTOMERS)
    order_items = pd.read_csv(ORDER_ITEMS)
    products = pd.read_csv(PRODUCTS)
    sellers = pd.read_csv(SELLERS)
    geolocation = pd.read_csv(GEOLOCATION)
    payments = pd.read_csv(PAYMENTS)
    reviews = pd.read_csv(REVIEWS)
    translation = pd.read_csv(TRANSLATION)
    
    df_dict = {
        'orders': orders,
        'customers': customers,
        'order_items': order_items,
        'products': products,
        'sellers': sellers,
        'geolocation': geolocation,
        'payments': payments,
        'reviews': reviews,
        'translation': translation,
    }
    return df_dict

df_dict = load_data()

# Data Cleaning Functions
def clean_null_values(df_dict):
    # Filling null values in the 'orders' dataset
    orders_df = df_dict.get('orders')
    if orders_df is not None:
        # For datetime columns: Fill based on order status
        orders_df['order_delivered_customer_date'] = orders_df.apply(
            lambda row: row['order_estimated_delivery_date'] if pd.isnull(row['order_delivered_customer_date']) 
            and row['order_status'] not in ['delivered', 'canceled'] else row['order_delivered_customer_date'], 
            axis=1
        )
        # For other columns, fill with 'Unknown'
        orders_df.fillna({
            'order_approved_at': 'Unknown',
            'order_delivered_carrier_date': 'Unknown',
        }, inplace=True)
        df_dict['orders'] = orders_df

    # Fill nulls in 'products' dataset
    products_df = df_dict.get('products')
    if products_df is not None:
        numeric_columns = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 
                           'product_name_lenght', 'product_description_lenght', 'product_photos_qty']
        products_df[numeric_columns] = products_df[numeric_columns].apply(lambda col: col.fillna(col.median()))
        products_df['product_category_name'].fillna('Unknown', inplace=True)
        df_dict['products'] = products_df

    # Drop columns with nulls in 'reviews' dataset
    reviews_df = df_dict.get('reviews')
    if reviews_df is not None:
        columns_with_nulls = reviews_df.columns[reviews_df.isnull().any()]
        reviews_cleaned = reviews_df.drop(columns=columns_with_nulls)
        df_dict['reviews'] = reviews_cleaned

def drop_duplicates(df_dict):
    for name, df in df_dict.items():
        df_cleaned = df.drop_duplicates()
        df_dict[name] = df_cleaned

def create_main_dataframe(df_dict):
    required_keys = ['orders', 'customers', 'order_items', 'products', 'reviews', 'geolocation', 'translation']
    for key in required_keys:
        if key not in df_dict:
            st.error(f"Dataset '{key}' is missing from df_dict")
            return None
    
    # Merge key datasets
    main_df = pd.merge(df_dict['orders'], df_dict['customers'], on='customer_id', how='inner')
    main_df = pd.merge(main_df, df_dict['order_items'], on='order_id', how='inner')
    main_df = pd.merge(main_df, df_dict['products'], on='product_id', how='inner')
    main_df = pd.merge(main_df, df_dict['reviews'], on='order_id', how='left')
    main_df = pd.merge(main_df, df_dict['geolocation'], 
                       left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    
    # Merge with the translation dataframe
    main_df = pd.merge(main_df, df_dict['translation'], on='product_category_name', how='left')
    main_df.drop(columns=['product_category_name'], inplace=True)
    main_df.rename(columns={'product_category_name_english': 'product_category_name'}, inplace=True)
    
    return main_df.sample(n=100000, random_state=42)

# Data Cleaning
clean_null_values(df_dict)
drop_duplicates(df_dict)
df = create_main_dataframe(df_dict)

st.success('Data loaded and cleaned successfully!')

if df is not None:
    def display_introduction(main_df):
        """
        Display an introduction with the number of rows (data points) and features (columns) in the dataset.
        """
        # Calculate the number of rows and columns in the main dataframe
        num_rows = main_df.shape[0]
        num_columns = main_df.shape[1]
        
        # Display the introduction
        st.subheader("Introduction")
        st.markdown(f"""
        This analysis is based on an extensive dataset provided by Olist, containing information about e-commerce orders in Brazil.

        - **Total Data Points (Rows)**: {num_rows}
        - **Total Features (Columns)**: {num_columns}

        We will explore various aspects of the dataset, including customer behavior, sales trends, delivery performance, and customer segmentation using RFM analysis.
        """)

    # Exploratory Data Analysis Functions
    def plot_order_status_distribution(main_df):
        st.subheader('1. Distribution of Order Status')
        status_counts = main_df['order_status'].value_counts().reset_index()
        status_counts.columns = ['Order Status', 'Count']
        fig = px.bar(status_counts, x='Order Status', y='Count', 
                     labels={'Order Status': 'Order Status', 'Count': 'Count'}, 
                     title='Order Status Distribution', text='Count')
        st.plotly_chart(fig)
        st.markdown(f"**Orders seem to be in check most of the time, with almost all being delivered. Some cases of orders not delivered seems to be unlikely or not recorded.**")

    def plot_avg_price_by_category(main_df):
        st.subheader('2. Average Price by Category')
        avg_price_category = main_df.groupby('product_category_name')['price'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(avg_price_category, x=avg_price_category.index, y=avg_price_category.values, 
                     labels={'x': 'Product Category', 'y': 'Average Price'}, 
                     title='Top 5 Highest Average Product Prices by Category', text=avg_price_category.values)
        st.plotly_chart(fig)
        st.markdown(f"**Computers and small home appliances among some of the highest pricing categories.**")

    def plot_delivery_time_trend(main_df):
        st.subheader('3. Average Delivery Time')
        main_df['order_purchase_timestamp'] = pd.to_datetime(main_df['order_purchase_timestamp'])
        main_df['order_delivered_customer_date'] = pd.to_datetime(main_df['order_delivered_customer_date'])
        valid_df = main_df.dropna(subset=['order_delivered_customer_date', 'order_purchase_timestamp']).copy()
        valid_df.loc[:, 'delivery_time_days'] = (valid_df['order_delivered_customer_date'] - valid_df['order_purchase_timestamp']).dt.days
        valid_df.loc[:, 'purchase_month'] = valid_df['order_purchase_timestamp'].dt.to_period('M').astype(str)
        avg_delivery_time = valid_df.groupby('purchase_month')['delivery_time_days'].mean().reset_index()
        fig = px.line(avg_delivery_time, x='purchase_month', y='delivery_time_days', 
                      labels={'purchase_month': 'Purchase Month', 'delivery_time_days': 'Average Delivery Time (Days)'}, 
                      title='Trend of Average Delivery Time Over Time')
        st.plotly_chart(fig)
        st.markdown(f"**Around late 2016, the service experienced a steep decrease in average delivery time. It has been generally stable ever since.**")

    def plot_revenue_proportion_by_category(main_df):
        st.subheader('4. Revenue Proportion by Category')
        main_df['revenue'] = main_df['price'] * main_df['order_item_id']
        revenue_by_category = main_df.groupby('product_category_name')['revenue'].sum().reset_index()
        top_10_categories = revenue_by_category.nlargest(10, 'revenue')
        revenue_by_category['product_category_name'] = revenue_by_category['product_category_name'].apply(
            lambda x: x if x in top_10_categories['product_category_name'].values else 'Other'
        )
        revenue_by_category = revenue_by_category.groupby('product_category_name')['revenue'].sum().reset_index()
        fig = px.pie(revenue_by_category, names='product_category_name', values='revenue', 
                     title='Proportion of Revenue by Product Category (Top 10 + Other)', 
                     labels={'product_category_name': 'Product Category', 'revenue': 'Total Revenue'})
        st.plotly_chart(fig)
        st.markdown(f"**Bed bath table and health beauty categories are among the highest selling products in the service.**")

    def plot_customer_demographic_scatter(main_df):
        st.subheader('5. Customer Demographic')
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-75, -35, -35, 6])  # Brazil's boundaries
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.scatter(main_df['geolocation_lng'], main_df['geolocation_lat'], color='red', alpha=0.9, s=1, transform=ccrs.PlateCarree())
        plt.title('Customer Demographic Scatter Plot in Brazil')
        st.pyplot(fig)
        st.markdown(f"**Most customers are from the eastern part of Brazil, specifically from metropolitan areas.**")

    # Business Questions Functions
    def plot_revenue_by_region(main_df):
        st.subheader('1. Which region produces the most revenue?')
        main_df['revenue'] = main_df['price'] * main_df['order_item_id']
        revenue_by_region = main_df.groupby('customer_state')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
        fig = px.bar(revenue_by_region, x='customer_state', y='revenue', 
                     labels={'customer_state': 'Customer State', 'revenue': 'Total Revenue'}, 
                     title='Revenue by Region (Customer State)', text='revenue')
        st.plotly_chart(fig)
        top_region = revenue_by_region.iloc[0]
        st.markdown(f"**The region that produces the most revenue is {top_region['customer_state']} with a total revenue of {top_region['revenue']:.2f}.**")

    def plot_avg_delivery_time_by_region(main_df):
        st.subheader('2. What is the average delivery time for each region?')
        main_df['order_delivered_customer_date'] = pd.to_datetime(main_df['order_delivered_customer_date'])
        main_df['order_purchase_timestamp'] = pd.to_datetime(main_df['order_purchase_timestamp'])
        main_df['delivery_time_days'] = (main_df['order_delivered_customer_date'] - main_df['order_purchase_timestamp']).dt.days
        avg_delivery_time_by_region = main_df.groupby('customer_state')['delivery_time_days'].mean().reset_index().sort_values(by='delivery_time_days', ascending=False)
        fig = px.bar(avg_delivery_time_by_region, x='customer_state', y='delivery_time_days', 
                     labels={'customer_state': 'Customer State', 'delivery_time_days': 'Average Delivery Time (Days)'}, 
                     title='Average Delivery Time by Region', text='delivery_time_days')
        st.plotly_chart(fig)
        st.markdown("**The average delivery time for each region has been plotted above.**")

    def count_late_deliveries(main_df):
        """
        Find if there are any late deliveries, how many, and plot late vs. not late deliveries.
        """
        # Convert relevant columns to datetime
        main_df['order_estimated_delivery_date'] = pd.to_datetime(main_df['order_estimated_delivery_date'])
        main_df['order_delivered_customer_date'] = pd.to_datetime(main_df['order_delivered_customer_date'])

        # Create a binary column to identify late deliveries
        main_df['late_delivery'] = main_df['order_delivered_customer_date'] > main_df['order_estimated_delivery_date']

        # Count late deliveries
        late_deliveries_count = main_df['late_delivery'].sum()
        not_late_deliveries_count = len(main_df) - late_deliveries_count

        # Display the late delivery count
        if late_deliveries_count > 0:
            st.markdown(f"**There are {late_deliveries_count} late deliveries recorded.**")
        else:
            st.markdown("**There are no late deliveries recorded.**")

        # Create a summary dataframe for plotting
        summary_df = pd.DataFrame({
            'Delivery Status': ['Not Late', 'Late'],
            'Count': [not_late_deliveries_count, late_deliveries_count]
        })

        # Plot the late vs not-late deliveries with Plotly
        fig = px.bar(summary_df, x='Delivery Status', y='Count', text='Count', 
                    title='Late vs Not Late Deliveries', 
                    labels={'Count': 'Number of Deliveries', 'Delivery Status': 'Delivery Status'},
                    color='Delivery Status')

        # Add labels to bars
        fig.update_traces(texttemplate='%{text}', textposition='outside')

        # Display the plot
        st.plotly_chart(fig)


    def analyze_causes_of_late_deliveries_correlation(main_df):
        st.subheader('4. What may be the cause of late deliveries?')
        main_df['is_late'] = (main_df['order_delivered_customer_date'] > main_df['order_estimated_delivery_date']).astype(int)
        numerical_columns = ['freight_value', 'price', 'product_weight_g', 'product_length_cm', 
                             'product_height_cm', 'product_width_cm', 'order_item_id']
        correlations = main_df[numerical_columns + ['is_late']].corr()['is_late'].drop('is_late').sort_values(ascending=False)
        fig = px.bar(correlations, x=correlations.values, y=correlations.index, orientation='h',
                     labels={'x': 'Correlation Coefficient', 'index': 'Features'},
                     title='Correlation with Late Deliveries (is_late)')
        st.plotly_chart(fig)
        st.markdown("""
**Freight Value, Product Weight, and Product Length are Positively Correlated with Late Deliveries**

- **Freight Value**: Higher freight values may indicate longer distances or special handling, leading to delays.
- **Product Weight**: Heavier items may require special logistics, increasing delivery time.
- **Product Length**: Larger dimensions can complicate shipping, causing delays.

**Order Item ID is Negatively Correlated with Late Deliveries**

- Larger orders might be prioritized, resulting in timely deliveries.
""")

    # RFM Analysis Functions
    def rfm_analysis(main_df):
        st.subheader('RFM (Recency, Frequency, Monetary) Analysis')
        # Add the markdown explanation for RFM
        st.markdown("""
        RFM analysis is a technique used in customer segmentation based on three key factors:
        
        - **Recency (R)**: How recently a customer made a purchase.
        - **Frequency (F)**: How often a customer makes a purchase.
        - **Monetary (M)**: How much money a customer spends.
        """)
        main_df['order_purchase_timestamp'] = pd.to_datetime(main_df['order_purchase_timestamp'])
        reference_date = main_df['order_purchase_timestamp'].max()
        rfm_table = main_df.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
            'order_id': 'count',
            'price': 'sum'
        }).reset_index()
        rfm_table.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']
        rfm_table['monetary'] = rfm_table['monetary'] * main_df.groupby('customer_unique_id')['order_item_id'].sum().values
        return rfm_table

    def plot_rfm_analysis(rfm_table):
        def categorize_rfm(rfm_table):
            rfm_table['Monetary_Category'] = pd.cut(rfm_table['monetary'],
                                                    bins=[-1, 100, 500, 1000, float('inf')],
                                                    labels=['Low Spender', 'Medium Spender', 'High Spender', 'Very High Spender'])
            rfm_table['Frequency_Category'] = pd.cut(rfm_table['frequency'],
                                                    bins=[-1, 1, 3, 5, float('inf')],
                                                    labels=['Rare', 'Occasional', 'Frequent', 'Very Frequent'])
            rfm_table['Recency_Category'] = pd.cut(rfm_table['recency'],
                                                bins=[-1, 30, 60, 120, float('inf')],
                                                labels=['Recent', 'Moderately Recent', 'Stale', 'Very Stale'])
            return rfm_table

        def add_count_and_percentage(ax, data, column_name):
            total = len(data)
            for p in ax.patches:
                count = int(p.get_height())
                percentage = f'{100 * count / total:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.text(x, y + 5, f'{count}\n({percentage})', ha='center', va='center')

        rfm_table = categorize_rfm(rfm_table)

        # Add explanations for the insights
        st.subheader('Recency Category Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Recency_Category', data=rfm_table, order=['Recent', 'Moderately Recent', 'Stale', 'Very Stale'], ax=ax)
        add_count_and_percentage(ax, rfm_table, 'Recency_Category')
        st.pyplot(fig)
        st.markdown("""
        - The fact that many customers fall into the **"Very Stale"** category suggests that while these customers may have been highly active in the past, they haven’t returned to the platform recently. This could indicate the need for re-engagement strategies, such as targeted email campaigns, special offers, or personalized recommendations to bring them back.
        """)

        st.subheader('Frequency Category Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Frequency_Category', data=rfm_table, order=['Rare', 'Occasional', 'Frequent', 'Very Frequent'], ax=ax)
        add_count_and_percentage(ax, rfm_table, 'Frequency_Category')
        st.pyplot(fig)
        st.markdown("""
        - Despite being stale in terms of recent purchases, these customers have made **frequent purchases** in the past. Customers who fall into the **"Very Frequent"** category are repeat buyers. This shows that although they are currently inactive, they were loyal and consistent buyers in the past. Retargeting them could yield strong results as they have demonstrated their willingness to engage repeatedly with the platform.
        """)

        st.subheader('Monetary Category Distribution')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Monetary_Category', data=rfm_table, order=['Low Spender', 'Medium Spender', 'High Spender', 'Very High Spender'], ax=ax)
        add_count_and_percentage(ax, rfm_table, 'Monetary_Category')
        st.pyplot(fig)
        st.markdown("""
        - These customers have spent a significant amount of money on the platform. A **"Very High Spender"** classification means that they have contributed a large share of revenue in the past. These high-spending customers are valuable to the business. Although they haven’t made a purchase recently, their past spending habits make them an attractive segment to target for future engagement.
        """)

    
    # Intro
    display_introduction(df)

    # Display EDA and Insights
    st.header('Exploratory Data Analysis')
    plot_order_status_distribution(df)
    plot_avg_price_by_category(df)
    plot_delivery_time_trend(df)
    plot_revenue_proportion_by_category(df)
    plot_customer_demographic_scatter(df)

    # Answering Business Questions
    st.header('Answering Business Questions')
    plot_revenue_by_region(df)
    plot_avg_delivery_time_by_region(df)
    count_late_deliveries(df)
    analyze_causes_of_late_deliveries_correlation(df)

    # RFM Analysis
    st.header('Customer Segmentation using RFM Analysis')
    rfm_table = rfm_analysis(df)
    plot_rfm_analysis(rfm_table)

else:
    st.error('Failed to create the main dataframe. Please check the data files.')

