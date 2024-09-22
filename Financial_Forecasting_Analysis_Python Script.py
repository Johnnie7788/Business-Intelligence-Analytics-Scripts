#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the financial dataset
df = pd.read_csv('Financial_Forecasting_Data.csv')

# Step 1: Handling Missing Data
def handle_missing_values(data):
    """Handle missing values using mean imputation for numeric columns."""
    numeric_data = data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    
    # Combine with non-numeric columns (if any)
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    data_clean = pd.concat([non_numeric_data.reset_index(drop=True), data_imputed], axis=1)
    
    return data_clean

df_clean = handle_missing_values(df)

# Step 2: Feature Engineering - Scaling the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_clean.drop(['Date'], axis=1)), 
                         columns=df_clean.drop(['Date'], axis=1).columns)

# Step 3: Split data for machine learning forecasting
# Assuming we are forecasting 'Seasonal_Revenue (EUR)'
X = scaled_df.drop(['Seasonal_Revenue (EUR)'], axis=1)
y = scaled_df['Seasonal_Revenue (EUR)']

# Step 4: Train-Test Split for forecasting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Financial Forecasting Model - Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate model performance
mae_rf = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (Random Forest): {mae_rf:.2f} EUR')

# Step 8: Time Series Forecasting using Holt-Winters Exponential Smoothing
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean.set_index('Date', inplace=True)

# Fit Holt-Winters model to forecast future seasonal revenue
model_holt = ExponentialSmoothing(df_clean['Seasonal_Revenue (EUR)'], trend='add', seasonal='add', seasonal_periods=12)
fit_holt = model_holt.fit()

# Forecast for the next 12 months
forecast_holt = fit_holt.forecast(steps=12)

# Fit ARIMA model to forecast future seasonal revenue
arima_model = ARIMA(df_clean['Seasonal_Revenue (EUR)'], order=(5,1,0))  # ARIMA(5,1,0) as an example
arima_fit = arima_model.fit()

# Forecast for the next 12 months using ARIMA
forecast_arima = arima_fit.forecast(steps=12)

# Step 9: Visualize the Results (including comparison between models)
def visualize_forecast(actual, forecast1, forecast2, title="Financial Forecast"):
    """Visualize actual vs forecasted values from two models."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values', marker='o')
    plt.plot(forecast1, label='Holt-Winters Forecast', marker='x')
    plt.plot(forecast2, label='ARIMA Forecast', marker='s')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Revenue (EUR)')
    plt.legend()
    plt.show()

# Visualize actual revenue and forecasts from both models
visualize_forecast(df_clean['Seasonal_Revenue (EUR)'][-24:], forecast_holt, forecast_arima, title="Revenue Forecast: Holt-Winters vs ARIMA")

# Step 10: Scenario Planning (What-If Analysis)
def scenario_planning(ad_spending_change, market_investment_change):
    """Simulate the effect of changes in ad spending and market investment on revenue."""
    adjusted_ad_spending = df_clean['Ad_Spending (EUR)'] * (1 + ad_spending_change)
    adjusted_market_investment = df_clean['Market_Investment (EUR)'] * (1 + market_investment_change)

    # Adjusting revenue by 10% of the changes in ad spending and market investment
    simulated_revenue = df_clean['Seasonal_Revenue (EUR)'] + 0.1 * (adjusted_ad_spending + adjusted_market_investment)

    # Visualize the impact of changes
    plt.figure(figsize=(10, 6))
    plt.plot(df_clean['Seasonal_Revenue (EUR)'], label='Original Revenue')
    plt.plot(simulated_revenue, label='Simulated Revenue (Ad Spend & Investment Adjusted)')
    plt.legend()
    plt.show()

# Example of Scenario Planning
scenario_planning(ad_spending_change=0.2, market_investment_change=0.1)

# Step 11: Add Financial Ratios and KPIs
df_clean['Profit Margin (%)'] = (df_clean['Profit (EUR)'] / df_clean['Revenue (EUR)']) * 100
df_clean['ROI (%)'] = (df_clean['Profit (EUR)'] / df_clean['Market_Investment (EUR)']) * 100

# Display the newly created financial ratios
print(df_clean[['Profit Margin (%)', 'ROI (%)']].describe())

# Step 12: Automatic Interpretation & Strategies
def automatic_interpreter_with_strategies(y_test, y_pred_rf, forecast_holt, forecast_arima, df):
    """Automatically interpret the forecasting results and provide detailed recommendations with strategies."""
    avg_actual = np.mean(y_test)
    avg_pred_rf = np.mean(y_pred_rf)
    avg_forecast_holt = np.mean(forecast_holt)
    avg_forecast_arima = np.mean(forecast_arima)
    
    # Calculate average Profit Margin and ROI
    avg_profit_margin = df['Profit Margin (%)'].mean()
    avg_roi = df['ROI (%)'].mean()
    
    print("\n--- Automatic Interpretation ---")
    print(f"Actual Average Seasonal Revenue: {avg_actual:.2f} EUR")
    print(f"Predicted Average Seasonal Revenue (Random Forest): {avg_pred_rf:.2f} EUR")
    print(f"Forecasted Average Revenue (Holt-Winters): {avg_forecast_holt:.2f} EUR")
    print(f"Forecasted Average Revenue (ARIMA): {avg_forecast_arima:.2f} EUR")
    
    # Recommendations based on model performance
    if avg_forecast_holt > avg_actual:
        print("Recommendation: The Holt-Winters model predicts revenue growth. Consider expanding business operations, increasing production, or investing in marketing campaigns.")
    elif avg_forecast_arima > avg_actual:
        print("Recommendation: ARIMA model shows potential growth. Focus on strategic investments and diversifying your product or service offerings to capitalize on this trend.")
    else:
        print("Recommendation: Both models indicate potential revenue decline. Prioritize cost-saving measures, optimize operational processes, and reassess your pricing strategy.")
    
    # Interpreting visualizations and scenario planning
    print("\n--- Visualizations Interpretation ---")
    print("Revenue Forecast: Both Holt-Winters and ARIMA show different trends in revenue forecasting.")
    print("Simulated Scenarios: A 20% increase in ad spending and 10% market investment growth could positively impact revenue by 10%.")

    # Profit Margin and ROI interpretation with strategies
    print(f"\n--- Financial Ratios ---")
    print(f"Average Profit Margin: {avg_profit_margin:.2f}%")
    print(f"Average ROI: {avg_roi:.2f}%")
    
    # Strategy suggestions for Profit Margin
    if avg_profit_margin > 15:
        print("Strategy: The company is maintaining a healthy profit margin. Focus on scaling operations while maintaining cost controls.")
    elif avg_profit_margin > 10:
        print("Strategy: Moderate profit margins. Improve operational efficiency and reduce waste to increase profit margins.")
    else:
        print("Strategy: Low profit margins. Reevaluate pricing strategy, cut non-essential expenses, and focus on high-margin products.")
    
    # Strategy suggestions for ROI
    if avg_roi > 10:
        print("Strategy: High ROI indicates strong investment returns. Consider reinvesting profits into scaling the business or exploring new ventures.")
    else:
        print("Strategy: Low ROI. Focus on improving the efficiency of capital allocation. Reassess current investments to identify underperforming assets.")

# Apply the interpreter for both models, including automatic strategies based on Profit Margin and ROI
automatic_interpreter_with_strategies(y_test, y_pred, forecast_holt, forecast_arima, df_clean)

# Step 13: Generate Professional Summary with Insights, Strategies, and Next Steps
def generate_summary_with_strategies(mae_rf, y_test, y_pred, forecast_holt, forecast_arima, df):
    """Generate a professional summary of the financial forecasting analysis with insights, strategies, and next steps."""
    avg_forecast_holt = np.mean(forecast_holt)
    avg_forecast_arima = np.mean(forecast_arima)
    avg_actual = np.mean(y_test)
    
    # Calculate average Profit Margin and ROI
    avg_profit_margin = df['Profit Margin (%)'].mean()
    avg_roi = df['ROI (%)'].mean()
    
    print("\n--- Professional Summary ---")
    print(f"Model Performance (Random Forest): Mean Absolute Error = {mae_rf:.2f} EUR")
    print(f"Average Actual Seasonal Revenue: {avg_actual:.2f} EUR")
    print(f"Average Forecasted Revenue (Holt-Winters, next 12 months): {avg_forecast_holt:.2f} EUR")
    print(f"Average Forecasted Revenue (ARIMA, next 12 months): {avg_forecast_arima:.2f} EUR")
    
    print(f"\nFinancial Ratios Overview:")
    print(f"Average Profit Margin: {avg_profit_margin:.2f}%")
    print(f"Average ROI: {avg_roi:.2f}%")
    
    # Business Outlook and Strategies
    if avg_forecast_holt > avg_actual:
        print("Business Outlook: Holt-Winters predicts steady revenue growth. Consider investing in expansion, product development, or hiring new staff.")
    elif avg_forecast_arima > avg_actual:
        print("Business Outlook: ARIMA forecasts revenue growth. Focus on capturing new market opportunities, optimizing sales strategies, and expanding brand presence.")
    else:
        print("Business Outlook: Both models indicate a potential downturn. Mitigate risks by reducing operational costs, re-evaluating supply chains, and securing long-term contracts.")
    
    # Detailed Strategy Suggestions
    print("\nStrategies:")
    
    if avg_profit_margin > 15:
        print("Strategy for Profit Margin: Continue scaling operations while ensuring cost control. Maintain efficient production and supply chain management.")
    elif avg_profit_margin > 10:
        print("Strategy for Profit Margin: Explore ways to optimize operations and reduce overhead costs. Focus on increasing operational efficiency.")
    else:
        print("Strategy for Profit Margin: Reevaluate pricing and focus on higher-margin products or services. Cut unnecessary expenses and streamline processes.")

    if avg_roi > 10:
        print("Strategy for ROI: High ROI means investments are generating solid returns. Reinvest profits into high-performing areas or consider expansion opportunities.")
    else:
        print("Strategy for ROI: Reassess capital allocation strategies. Focus on improving asset utilization and reallocating resources to better-performing areas.")
    
    print("\nNext Steps:")
    print("- Continuously monitor financial performance and adjust your strategy based on quarterly trends.")
    print("- Reassess investment strategies and identify opportunities for better ROI.")
    print("- Focus on improving profit margins through cost management, pricing adjustments, and operational efficiencies.")
    print("- Perform regular scenario planning to ensure you're prepared for various market conditions.")

# Generate the detailed professional summary with strategies and next steps
generate_summary_with_strategies(mae_rf, y_test, y_pred, forecast_holt, forecast_arima, df_clean)


