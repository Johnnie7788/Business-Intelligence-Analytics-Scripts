#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_squared_error

# ---- Step 1: Read the dataset 'pricing_modeling_data.csv' ----
df = pd.read_csv("pricing_modeling_data.csv")

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# ---- Step 2: Price Adjustments based on Sentiment, Supply-Demand, Risk ----
# Sentiment-Adjusted Price (SAPF)
df["Sentiment_Adjusted_Price"] = df["Base_Price"] * (1 + df["Sentiment_Score"])

# Supply-Demand Elasticity Adjusted Price (FSDEM)
df["Supply_Demand_Adjusted_Price"] = df["Base_Price"] * (df["Demand"] / df["Supply"])

# Risk-Adjusted Price (RAPSA)
df["Risk_Adjusted_Price"] = df["Base_Price"] * (1 + df["Risk_Factor"])

# Final Price: Combine all effects (Sentiment, Supply-Demand, Risk)
df["Final_Price"] = (df["Sentiment_Adjusted_Price"] + df["Supply_Demand_Adjusted_Price"] + df["Risk_Adjusted_Price"]) / 3

# ---- Step 3: Random Forest Regression for Multi-Variable Prediction ----
# Prepare the features (X) and target (y)
X = df[['Supply', 'Demand', 'Sentiment_Score', 'Risk_Factor']]
y = df['Final_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict prices using the test set
y_pred_rf = rf_model.predict(X_test)

# Print Feature Importance
importances = rf_model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f'Feature: {feature}, Importance: {importance:.4f}')

# Evaluate the Random Forest model's accuracy using Mean Squared Error
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (Random Forest): {mse_rf:.2f}")

# ---- Step 4: Prophet for Time Series Forecasting ----
# Prepare data for Prophet (requires 'ds' for dates and 'y' for prices)
df_prophet = df[['Date', 'Final_Price']].rename(columns={'Date': 'ds', 'Final_Price': 'y'})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df_prophet)

# Make future predictions for the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot the forecasted final prices
model.plot(forecast)
plt.title('Forecasted Final Price Over Time')
plt.show()

# ---- Step 5 ----
df.to_csv("pricing_modeling_data_updated.csv", index=False)

# ---- Step 6: Visualizations (for further insights) with Interpreter ----

# Visualization 1: Sentiment-Adjusted Price over time (SAPF)
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Sentiment_Adjusted_Price', data=df, hue='Region')
plt.title('Sentiment-Adjusted Price Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Sentiment-Adjusted Price (€)')
plt.xticks(rotation=45)
plt.show()

# Interpretation for Visualization 1
avg_sentiment_price = df.groupby('Region')['Sentiment_Adjusted_Price'].mean()
print("---- Interpretation for Sentiment-Adjusted Price ----")
for region, price in avg_sentiment_price.items():
    print(f"Region: {region}, Average Sentiment-Adjusted Price: {price:.2f} €")
    if price > df['Base_Price'].mean():
        print(f"Recommendation: Sentiment in {region} is driving prices higher. Consider hedging strategies.")
    else:
        print(f"Recommendation: Sentiment in {region} is moderate. Focus on maintaining current pricing strategy.")

# Visualization 2: Supply-Demand Elasticity Adjusted Price (FSDEM) by Region
plt.figure(figsize=(10,6))
sns.barplot(x='Region', y='Supply_Demand_Adjusted_Price', data=df, hue='Cluster')
plt.title('Supply-Demand Elasticity Adjusted Price by Region')
plt.xlabel('Region')
plt.ylabel('Supply-Demand Adjusted Price (€)')
plt.show()

# Interpretation for Visualization 2
avg_supply_demand_price = df.groupby('Region')['Supply_Demand_Adjusted_Price'].mean()
print("---- Interpretation for Supply-Demand Elasticity Adjusted Price ----")
for region, price in avg_supply_demand_price.items():
    print(f"Region: {region}, Average Supply-Demand Adjusted Price: {price:.2f} €")
    if price > df['Base_Price'].mean():
        print(f"Recommendation: {region} is experiencing high demand relative to supply. Consider increasing supply to stabilize prices.")
    else:
        print(f"Recommendation: {region} has balanced supply-demand conditions. Monitor for future volatility.")

# Visualization 3: Risk-Adjusted Price (RAPSA) vs. Risk Factor
plt.figure(figsize=(10,6))
sns.scatterplot(x='Risk_Factor', y='Risk_Adjusted_Price', data=df, hue='Region', size='Risk_Factor', sizes=(20, 200))
plt.title('Risk-Adjusted Price vs. Risk Factor by Region')
plt.xlabel('Risk Factor')
plt.ylabel('Risk-Adjusted Price (€)')
plt.show()

# Interpretation for Visualization 3
high_risk_regions = df[df['Risk_Factor'] > 0.2]['Region'].unique()
print("---- Interpretation for Risk-Adjusted Price ----")
if len(high_risk_regions) > 0:
    print(f"Regions with high risk factors: {', '.join(high_risk_regions)}")
    print("Recommendation: Focus on risk mitigation strategies in these regions. Consider insurance or diversifying sources of power.")
else:
    print("All regions have moderate or low risk. Continue with current strategies.")

# Visualization 4: Final Price across different Clusters (MLFC)
plt.figure(figsize=(10,6))
sns.boxplot(x='Cluster', y='Final_Price', data=df)
plt.title('Final Price Distribution by Market Cluster')
plt.xlabel('Market Cluster')
plt.ylabel('Final Price (€)')
plt.show()

# Interpretation for Visualization 4
cluster_price_stats = df.groupby('Cluster')['Final_Price'].mean()
print("---- Interpretation for Final Price by Market Cluster ----")
for cluster, price in cluster_price_stats.items():
    print(f"Cluster: {cluster}, Average Final Price: {price:.2f} €")
    if cluster == 'High-Demand':
        print(f"Recommendation: Consider strategic price increases in {cluster} clusters to capitalize on demand.")
    elif cluster == 'Low-Demand':
        print(f"Recommendation: Focus on demand generation strategies (e.g., marketing) in {cluster} clusters.")

