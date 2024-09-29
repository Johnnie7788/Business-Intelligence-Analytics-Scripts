#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')  # To ignore warnings for a cleaner output

# This script can be integrated with AWS, Azure, or Google Cloud for data storage, scalability, and integration with cloud-based AI/ML services.

# Step 1: Data Preprocessing for Supply Chain Analytics (Handling Missing Values and Data Imbalance using SMOTE)
def preprocess_data(df):
    print("Columns in the dataset:", df.columns)
    
    # Handling missing values
    # For numerical columns, replace missing values with the median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))
    
    # For categorical columns, replace missing values with the mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Encode categorical columns into numeric format (Label Encoding)
    label_encoder = LabelEncoder()
    for col in ['Supplier', 'Location', 'Customer Type', 'Product']:
        df[col] = label_encoder.fit_transform(df[col])

    # Addressing data imbalance using SMOTE for the 'Customer Type' column
    smote = SMOTE(random_state=42)

    # Drop 'Date' before applying SMOTE
    X = df.drop(['Customer Type', 'Date'], axis=1)  # Features (excluding target and 'Date')
    y = df['Customer Type']  # Target

    # Applying SMOTE to the features and target
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Creating the final balanced dataframe (re-add 'Date' after SMOTE)
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Customer Type'])], axis=1)
    df_resampled['Date'] = df['Date'].reset_index(drop=True)  # Re-add 'Date' column after resampling

    return df_resampled

# Step 2: Dynamic Demand Forecasting with Adjustable Forecast Window
def forecast_demand(df, forecast_days=90):
    # Handle missing values in the 'Date' column
    df = df.dropna(subset=['Date'])  # Drop rows where 'Date' is missing
    
    df_forecast = df[['Date', 'Units Sold']].rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    
    model = Prophet()
    model.fit(df_forecast)
    
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    model.plot(forecast)
    plt.title(f'Demand Forecast for Next {forecast_days} Days')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.show()

    print(f"\nDemand Forecast Interpretation for the next {forecast_days} days:")
    print(f"- The forecast predicts demand for the next {forecast_days} days based on historical data.")
    print("- If the forecast shows increasing demand, you may need to adjust inventory levels to avoid stockouts.")
    print("- Conversely, if demand is predicted to drop, you can reduce inventory levels to avoid excess stock.")

# Step 3: Enhanced Inventory Optimization Using Random Forest with Hyperparameter Tuning
def optimize_inventory(df):
    X = df[['Inventory Level', 'Lead Time (Days)', 'Supplier Rating (Scale of 1-5)', 'Price per Unit (EUR)', 'Location', 'Customer Type']]
    y = df['Units Sold']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Improvement 1: Hyperparameter tuning using RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    # RandomizedSearchCV to find best hyperparameters
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    
    rf_random.fit(X_train, y_train)
    
    # Make predictions with the best model
    y_pred = rf_random.best_estimator_.predict(X_test)
    
    # Improvement 2: Include more metrics like R-squared and Mean Squared Error
    error = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Absolute Error for Inventory Optimization: {error:.2f} units")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")

    print("\nInventory Optimization Interpretation:")
    if error < 800:
        print("- The model's error is low, meaning that it provides accurate predictions of future inventory needs.")
        print("- You can use this model to optimize your stock levels, reducing the risk of overstocking or stockouts.")
    else:
        print("- The model's error is relatively high, indicating that predictions might need further tuning.")
        print("- Consider refining the model with additional data or adjusting the features used for prediction.")
    
    return rf_random.best_estimator_, y_test, y_pred

# Step 4: Advanced Supplier Performance Clustering with Risk Profiling
def supplier_performance_analysis(df):
    X = df[['Supplier Rating (Scale of 1-5)', 'Delivery Time (Days)', 'Defective Rate (%)', 'Location']]
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Supplier Cluster'] = kmeans.fit_predict(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Supplier Rating (Scale of 1-5)', y='Defective Rate (%)', hue='Supplier Cluster', data=df, palette='Set1')
    plt.title('Supplier Performance Clusters')
    plt.xlabel('Supplier Rating (Scale of 1-5)')
    plt.ylabel('Defective Rate (%)')
    plt.legend(loc='upper right')
    plt.show()
    
    print("\nSupplier Performance and Risk Profiling Interpretation:")
    for cluster in sorted(df['Supplier Cluster'].unique()):
        cluster_data = df[df['Supplier Cluster'] == cluster]
        avg_rating = cluster_data['Supplier Rating (Scale of 1-5)'].mean()
        avg_defective_rate = cluster_data['Defective Rate (%)'].mean()
        avg_delivery_time = cluster_data['Delivery Time (Days)'].mean()
        print(f"Cluster {cluster}: Avg Supplier Rating = {avg_rating:.2f}, Avg Defective Rate = {avg_defective_rate:.2f}%, Avg Delivery Time = {avg_delivery_time:.2f} days")
        
        if avg_defective_rate < 2 and avg_delivery_time < 10:
            print(f"- Cluster {cluster} represents high-performing suppliers with low risk. You should prioritize working with suppliers in this cluster.")
        else:
            print(f"- Cluster {cluster} represents underperforming suppliers with higher risks. Consider renegotiating contracts or switching suppliers in this cluster.")
    
# Step 5: Risk Mitigation and Supplier Strategy Recommendations
def supply_chain_risk_mitigation(df):
    high_risk_suppliers = df[(df['Defective Rate (%)'] > 2) | (df['Lead Time (Days)'] > 12)]
    
    print("\nHigh-Risk Suppliers Identified:")
    print(high_risk_suppliers[['Supplier', 'Defective Rate (%)', 'Lead Time (Days)']])
    
    print("\nRecommendations for Risk Mitigation:")
    for supplier, defective_rate, lead_time in zip(high_risk_suppliers['Supplier'], high_risk_suppliers['Defective Rate (%)'], high_risk_suppliers['Lead Time (Days)']):
        if defective_rate > 2:
            print(f"- {supplier} has a high defective rate of {defective_rate:.2f}%. You should consider improving quality control or finding alternative suppliers.")
        if lead_time > 12:
            print(f"- {supplier} has long lead times (over {lead_time} days). Consider improving logistics or switching to a faster supplier.")

# Step 6: Automated Supply Chain Summary with Real-Time Insights
def generate_summary(df, y_test, y_pred):
    total_suppliers = df['Supplier'].nunique()
    high_performers = len(df[df['Supplier Cluster'] == 0])
    underperformers = len(df[df['Supplier Cluster'] == 3])
    
    print("\nProfessional Summary:")
    print(f"- Total Suppliers Analyzed: {total_suppliers}")
    print(f"- High-Performing Suppliers: {high_performers}")
    print(f"- Underperforming Suppliers: {underperformers}")
    
    print("\nKey Recommendations:")
    print(f"- Focus on maintaining strong relationships with high-performing suppliers (Cluster 0) to ensure supply chain reliability.")
    print(f"- For underperforming suppliers (Cluster 3), consider switching suppliers, renegotiating contracts, or improving quality control processes.")
    print(f"- Utilize the demand forecasting model to better align inventory levels with expected customer demand, reducing both overstock and stockouts.")
    print(f"- The inventory optimization model shows a mean absolute error of {mean_absolute_error(y_test, y_pred):.2f} units, indicating that it can significantly improve stock management.")
    print(f"- Consider refining models further by incorporating additional data or features such as supplier cost, logistics, or seasonal trends.")
    print(f"\nConclusion:")
    print(f"- This analysis provides actionable insights across multiple areas of your supply chain, including demand forecasting, inventory management, and supplier performance.")
    print(f"- Implementing these recommendations can lead to significant improvements in efficiency, cost management, and supply chain resilience.")

# Load the dataset and run the analysis
df = pd.read_csv('supply_chain_data.csv')
df = preprocess_data(df)

# Run forecasts and models
forecast_demand(df, forecast_days=120)
best_model, y_test, y_pred = optimize_inventory(df)
supplier_performance_analysis(df)
supply_chain_risk_mitigation(df)
generate_summary(df, y_test, y_pred)

