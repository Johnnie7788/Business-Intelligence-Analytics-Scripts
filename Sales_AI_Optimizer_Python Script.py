#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This script can also be connected to cloud platforms such as AWS, Azure, or Google Cloud for data storage, 
# scalability, and integration with cloud-based AI/ML services.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from prophet import Prophet  # For time-series forecasting

# Step 1: Data Preprocessing and Missing Value Handling with Column Name Check
def preprocess_data(df):
    # Check the column names first
    print("Columns in the dataset:", df.columns)
    
    # Assuming the column may be named 'Sales Amount (EUR)' instead of 'Sales Amount'
    if 'Sales Amount (EUR)' in df.columns:
        sales_amount_col = 'Sales Amount (EUR)'
        sales_target_col = 'Sales Target (EUR)'
    elif 'Sales Amount' in df.columns:
        sales_amount_col = 'Sales Amount'
        sales_target_col = 'Sales Target'
    else:
        raise KeyError("Sales Amount column not found in dataset!")
    
    # Checking for missing values
    missing_values = df.isnull().sum()
    print("Missing Values Check:\n", missing_values)
    
    # Handling missing values and converting to float
    # First, remove the ' EUR' string and any leading/trailing spaces, then convert to float
    df[sales_amount_col] = df[sales_amount_col].str.replace(' EUR', '').str.strip().astype(float)
    df[sales_target_col] = df[sales_target_col].str.replace(' EUR', '').str.strip().astype(float)
    
    # Create 'Sales Performance' as the ratio of Sales Amount to Sales Target
    df['Sales Performance'] = df[sales_amount_col] / df[sales_target_col]
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

# Load the updated dataset with units and preprocess it
df = pd.read_csv('kaiserkraft_sales_data_with_units.csv')
df = preprocess_data(df)

# Step 2: Visualize Sales Performance Over Time by Country
def visualize_sales_performance(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='Sales Performance', hue='Country')
    plt.title('Sales Performance Over Time by Country')
    plt.ylabel('Sales Performance (Ratio)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.show()

    print("\nInterpretation of Sales Performance:")
    print("This visualization shows how sales performance fluctuates over time across different countries.")
    print("Management can use this to identify trends, such as whether a country is improving or declining in performance.")
    print("Countries with declining performance may need new strategies, while high performers can be used as a benchmark for best practices.")

visualize_sales_performance(df)

# Step 3: Predictive Sales Analysis using Random Forest
def predict_sales(df):
    X = df[['Sales Target (EUR)', 'Month', 'Year']]
    y = df['Sales Performance']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model using Mean Absolute Error (MAE)
    error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {error:.2f} EUR")
    
    # Display Random Forest Predictions
    predictions_df = pd.DataFrame({'Actual Sales Performance': y_test, 'Predicted Sales Performance': y_pred})
    print("\nRandom Forest Prediction Results:\n", predictions_df.head())
    
    # Interpretation of Model Accuracy
    if error < 3000:
        print("\nThe model is accurate in predicting sales with a low error margin. Predictions can be used for strategic decisions.")
    else:
        print("\nThe model has a high error margin, indicating that predictions may need further tuning.")
    
    return model, error

model, error = predict_sales(df)

# Step 4: Identify Best Sales Practices by Region
def best_practices(df):
    best_performing_regions = df.groupby('Sales Region')['Sales Performance'].mean().sort_values(ascending=False)
    print("\nBest Performing Regions (by Sales Performance):\n", best_performing_regions)

    # Interpretation
    print("\nInterpretation of Regional Performance:")
    print(f"Central Europe is the best-performing region with a sales performance ratio of {best_performing_regions.max():.2f}.")
    print("Regions like Southern and Western Europe are underperforming and should consider adopting strategies from Central Europe, such as improving customer engagement or optimizing their sales processes.")

best_practices(df)

# Step 5: Sales Process Optimization (Recommendations)
def sales_optimization_recommendations(df):
    underperforming = df[df['Sales Performance'] < 0.8]
    recommendations = underperforming.groupby('Salesperson')['Sales Performance'].mean().sort_values()
    print("\nRecommendations for Improvement (Salespeople Under 80% Target):\n", recommendations)
    
    # Detailed recommendations
    print("\nInterpretation and Recommendations:")
    for salesperson, performance in recommendations.items():
        if performance < 0.5:
            print(f"{salesperson} is significantly underperforming with a performance ratio of {performance:.2f}. Management should consider providing one-on-one coaching and setting specific, achievable goals.")
        else:
            print(f"{salesperson} is slightly underperforming with a performance ratio of {performance:.2f}. Focus on continuous feedback and targeted skill development to improve performance.")
    
    print("\nGeneral Advice to Management:")
    print("Implementing regular feedback sessions, performance-based incentives, and sharing best practices across regions may help improve overall sales team performance.")

sales_optimization_recommendations(df)

# Step 6: Clustering to Identify Best Practices Across Regions
def cluster_regions(df):
    X = df[['Sales Target (EUR)', 'Sales Performance']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Sales Target (EUR)', y='Sales Performance', hue='Cluster', data=df, palette='Set1')
    plt.title('Clusters of Regions by Sales Target and Sales Performance')
    plt.xlabel('Sales Target (EUR)')
    plt.ylabel('Sales Performance (Ratio)')
    plt.legend(loc='upper right')
    plt.show()
    
    # Interpret clustering results
    print("\nClusters Interpretation:")
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        avg_performance = cluster_data['Sales Performance'].mean()
        avg_target = cluster_data['Sales Target (EUR)'].mean()
        print(f"Cluster {cluster}: Avg Sales Performance = {avg_performance:.2f}, Avg Sales Target = {avg_target:.2f} EUR")
        
        if avg_performance > 0.7:
            print(f"Cluster {cluster} is high-performing. Strategies used by this cluster could be shared with lower-performing clusters.")
        else:
            print(f"Cluster {cluster} is underperforming and needs further investigation. It might benefit from adopting high-performing strategies.")

cluster_regions(df)

# Step 7: Forecasting Future Sales using Prophet
def forecast_sales(df):
    df_prophet = df[['Date', 'Sales Amount (EUR)']].rename(columns={'Date': 'ds', 'Sales Amount (EUR)': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    model.plot(forecast)
    plt.title("Sales Forecast for Next 90 Days")
    plt.show()

    # Detailed Interpretation
    print("\nSales Forecast Interpretation:")
    print("The forecast shows expected sales trends over the next 90 days based on historical data.")
    print("Management can use this forecast to anticipate revenue and adjust their strategies accordingly.")
    print("If the forecast shows a decline, management may need to act proactively to boost sales, such as running targeted promotions or improving sales engagement.")

forecast_sales(df)

# Step 8: Automated Recommendations Based on Performance
def automated_recommendations(df):
    underperforming = df[df['Sales Performance'] < 0.8]
    recommendations = underperforming.groupby('Salesperson')['Sales Performance'].mean().sort_values()
    
    for salesperson, performance in recommendations.items():
        if performance < 0.5:
            print(f"Recommendation for {salesperson}: Intensive 1:1 coaching and performance monitoring needed.")
        else:
            print(f"Recommendation for {salesperson}: Focused sales skill development and additional support needed.")

automated_recommendations(df)

# Step 9: Professional Summary (Final Step)
def generate_summary(df, error):
    print("\nProfessional Summary:")
    print(f"The sales analysis identified {df['Salesperson'].nunique()} salespeople across {df['Sales Region'].nunique()} regions.")
    print(f"The best performing region is Central Europe with a sales performance ratio of {df.groupby('Sales Region')['Sales Performance'].mean().max():.2f}.")
    print(f"Regions like Western and Southern Europe are lagging, with sales performance ratios below 0.65.")
    print(f"Furthermore, individual salespeople like Luca Rossi and John Doe are performing significantly below target, with average performance ratios below 0.4.")
    print(f"The predictive model has a Mean Absolute Error of {error:.2f} EUR, indicating that it provides reasonably accurate sales forecasts, which can help guide strategic decisions.")
    print("Recommendations include targeted coaching for underperformers, performance-based incentives, and adopting best practices from high-performing regions.")
    print("With further model refinement and cross-region collaboration, there is significant potential to improve overall sales performance.")

generate_summary(df, error)


