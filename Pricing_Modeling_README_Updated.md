
# Pricing Modeling Script

## Overview
This repository provides a script that calculates and predicts prices by incorporating key variables such as Sentiment, Supply-Demand Elasticity, and Risk Factors. The script employs both Random Forest Regression for multi-variable prediction and Prophet for time series forecasting of future prices. It also includes detailed visualizations and interpretations of the price modeling process, offering actionable insights for businesses.

Key features include:
1. **Sentiment-Adjusted Pricing**: Adjusts base price according to customer or market sentiment.
2. **Supply-Demand Elasticity Adjustments**: Accounts for changes in demand and supply to adjust pricing.
3. **Risk-Adjusted Pricing**: Adjusts prices based on external risk factors to reflect market uncertainty.
4. **Random Forest Regression**: A machine learning model used to predict prices based on sentiment, supply, demand, and risk.
5. **Time Series Forecasting**: The Prophet model is used to forecast prices for the next 30 days.
6. **Visualizations & Business Insights**: Provides graphical insights into sentiment, supply-demand balance, and risk, along with business recommendations based on the model's outputs.

## Key Features & Functionality

### 1. Data Loading and Preprocessing
The script starts by loading the dataset pricing_modeling_data.csv. The dataset is complete with no missing values, so no additional imputation or preprocessing steps are required for missing data.

### 2. Sentiment-Adjusted Price Calculation
The Sentiment-Adjusted Price (SAPF) is calculated by adjusting the base price based on the Sentiment Score, which ranges from -1 (negative sentiment) to 1 (positive sentiment). This adjustment helps identify how changes in customer or market sentiment affect pricing.

### 3. Supply-Demand Elasticity Adjusted Price
The script calculates the Supply-Demand Elasticity Adjusted Price (FSDEM), which reflects the effects of supply and demand fluctuations on pricing. Prices are adjusted upward when demand exceeds supply and downward when supply exceeds demand.

### 4. Risk-Adjusted Price
Prices are further adjusted based on Risk Factors. The Risk-Adjusted Price (RAPSA) incorporates external risks, such as geopolitical events or market instability, which may influence pricing decisions.

### 5. Final Price Calculation
The final price is calculated as an average of the Sentiment-Adjusted Price, Supply-Demand Elasticity Adjusted Price, and Risk-Adjusted Price. This provides a holistic view of the factors influencing the final pricing strategy.

### 6. Random Forest Regression for Multi-Variable Prediction
A Random Forest Regression model is employed to predict final prices based on variables such as Supply, Demand, Sentiment Score, and Risk Factor. The model's performance is evaluated using Mean Squared Error (MSE), and the script reports feature importance to indicate the impact of each variable on price prediction.

### 7. Prophet Time Series Forecasting
The script uses the Prophet model to forecast future prices for the next 30 days based on historical data. This helps businesses understand potential future price trends and plan their pricing strategies accordingly.

### 8. Data Visualization & Interpretation
The script generates visualizations to offer deeper insights into the price modeling process:
1. **Sentiment-Adjusted Price Over Time**: A line plot showing how sentiment affects pricing over time across different regions.
2. **Supply-Demand Elasticity Adjusted Price by Region**: A bar plot displaying the supply-demand adjusted prices by region and market cluster.
3. **Risk-Adjusted Price vs. Risk Factor**: A scatter plot illustrating the relationship between risk factors and adjusted prices across regions.
4. **Final Price Distribution by Market Cluster**: A box plot visualizing the distribution of final prices across different market clusters.

Each visualization includes automated interpretations and business recommendations to guide strategic decision-making.

## Dataset
This includes the first 5 rows of the pricing modeling dataset. This shows the structure of the data used in the script, which includes:

- **Date**: The date of the pricing event.
- **Region**: The region where the pricing event occurs (e.g., Central Europe, Western Europe, etc.).
- **Sentiment_Score**: A numerical score representing market sentiment.
- **Supply**: The available supply of a product or service.
- **Demand**: The demand for the product or service.
- **Risk_Factor**: A risk score reflecting external market risks.
- **Cluster**: The market cluster (e.g., High-Demand, Moderate-Demand, Low-Demand).
- **Base_Price**: The base price of the product or service before any adjustments.
- **Sentiment_Adjusted_Price**: The price adjusted based on the sentiment score.
- **Supply_Demand_Adjusted_Price**: The price adjusted based on the supply-demand relationship.
- **Risk_Adjusted_Price**: The price adjusted based on external risk factors.
- **Final_Price**: The final price, calculated as the average of sentiment-adjusted, supply-demand adjusted, and risk-adjusted prices.

## How to Use

### 1. Install Required Dependencies
Ensure the required libraries are installed.

### 2. Prepare Your Dataset
Ensure your dataset is named pricing_modeling_data.csv and is located in the root directory of the project. The script is designed to handle CSV files but can be adapted for other formats.

### 3. Run the Script
Once the dataset is prepared, execute the script.

### 4. Analyze the Output
The script will generate:

- Predicted final prices based on multi-variable analysis.
- Future price forecasts for the next 30 days.
- Detailed visualizations of sentiment, supply-demand balance, and risk-adjusted prices.
- Business recommendations based on the generated visualizations and predictions.

### 5. Customize the Script
The script is flexible and can be easily adapted for different datasets, models, and business scenarios. You can adjust the parameters, features, or add more data sources depending on your specific needs.

## Libraries Used
- **Pandas**: For data manipulation and preprocessing.
- **Numpy**: For numerical computations and sample data generation.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For building and evaluating the Random Forest regression model.
- **Prophet**: For time series forecasting of future prices.

## Contribution
Contributions are welcome!

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.
