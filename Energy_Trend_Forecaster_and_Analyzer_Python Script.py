#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# 1. Reading Data from CSV with Error Handling
def read_energy_data(csv_file):
    """
    Reads energy data from a CSV file with error handling.
    """
    try:
        df = pd.read_csv(csv_file)
        logging.info("Data successfully read from CSV.")
        return df
    except FileNotFoundError:
        logging.error(f"File {csv_file} not found.")
    except Exception as e:
        logging.error(f"Error reading data: {e}")
    return None

# 2. Monitoring and Analyzing Global Energy Demand Trends
def analyze_energy_trends(df):
    """
    Plots energy demand trends over time and handles large datasets efficiently.
    Automatically interprets the trend and makes professional recommendations.
    """
    try:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Efficient plot for large datasets
        plt.figure(figsize=(10, 5))
        plt.plot(df['energy_demand (GWh)'], marker='o', linestyle='-')
        plt.title("Energy Demand Trends Over Time")
        plt.xlabel("Date")
        plt.ylabel("Energy Demand (GWh)")
        plt.grid(True)
        plt.show()
        
        # Interpretation
        trend = df['energy_demand (GWh)'].pct_change().mean()
        if trend > 0:
            print("Interpretation: The energy demand is increasing on average over time.")
            print("Recommendation: Prepare for increased capacity requirements and consider investments in renewable energy sources to meet growing demand.")
        elif trend < 0:
            print("Interpretation: The energy demand is decreasing on average over time.")
            print("Recommendation: Investigate the reasons for declining demand (e.g., energy efficiency, economic factors) and adjust supply accordingly.")
        else:
            print("Interpretation: The energy demand is stable over time.")
            print("Recommendation: Maintain current energy production levels, but keep an eye on external factors that could influence future demand.")
    except Exception as e:
        logging.error(f"Error analyzing energy trends: {e}")

# 3. Analyzing Fundamental Drivers
def analyze_fundamentals(df):
    """
    Analyzes the impact of macroeconomic indicators on energy demand.
    Automatically interprets the results and makes recommendations.
    """
    try:
        X = df[['GDP_growth (%)', 'technology_cost_reduction (%)']]
        y = df['energy_demand (GWh)']
        model = LinearRegression().fit(X, y)
        print("Linear Regression Model Coefficients:")
        print(f"GDP Growth Coefficient: {model.coef_[0]:.2f}")
        print(f"Technology Cost Reduction Coefficient: {model.coef_[1]:.2f}")
        
        # Interpretation
        if model.coef_[0] > 0:
            print("Interpretation: GDP growth positively impacts energy demand.")
            print("Recommendation: As the economy grows, expect higher energy consumption. Consider scaling energy production accordingly.")
        else:
            print("Interpretation: GDP growth negatively impacts energy demand.")
            print("Recommendation: Investigate why economic growth is leading to lower energy demand. It could be due to energy-efficient technologies or shifts in energy consumption patterns.")
        
        if model.coef_[1] < 0:
            print("Interpretation: Reductions in technology costs are leading to lower energy demand.")
            print("Recommendation: As technology becomes cheaper and more efficient, energy consumption might decrease. Consider adjusting pricing models or exploring new energy-efficient technologies.")
        else:
            print("Interpretation: Reductions in technology costs are leading to higher energy demand.")
            print("Recommendation: Consider leveraging new, affordable technologies to increase energy supply to meet rising demand.")
    except Exception as e:
        logging.error(f"Error analyzing fundamental drivers: {e}")

# 4. Enhancing Energy Demand Scenarios
def forecast_energy_demand(df):
    """
    Forecasts future energy demand using Prophet.
    Automatically interprets the forecast and makes recommendations.
    """
    try:
        df_prophet = df.reset_index().rename(columns={'date': 'ds', 'energy_demand (GWh)': 'y'})
        model = Prophet()
        model.fit(df_prophet[['ds', 'y']])
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        model.plot(forecast)
        plt.title("Energy Demand Forecast")
        plt.xlabel("Date")
        plt.ylabel("Energy Demand (GWh)")
        plt.show()
        
        # Interpretation
        demand_forecast_mean = forecast['yhat'].pct_change().mean()
        if demand_forecast_mean > 0:
            print("Interpretation: The forecast indicates an upward trend in energy demand.")
            print("Recommendation: Plan for capacity expansion and consider investments in scalable, renewable energy solutions to meet future demand.")
        elif demand_forecast_mean < 0:
            print("Interpretation: The forecast indicates a downward trend in energy demand.")
            print("Recommendation: Reassess energy supply strategies and explore opportunities for increasing energy efficiency or reducing costs.")
        else:
            print("Interpretation: The forecast suggests stable energy demand.")
            print("Recommendation: Maintain current production levels, but monitor for potential shifts in demand due to external factors.")
    except Exception as e:
        logging.error(f"Error forecasting energy demand: {e}")

# 5. Forecasting Fuel Commodity Prices
def forecast_fuel_prices(df):
    """
    Forecasts future fuel prices using ARIMA model.
    Automatically interprets the forecast and makes recommendations.
    """
    try:
        model = ARIMA(df['fuel_price (USD/barrel)'], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['fuel_price (USD/barrel)'], label="Historical Fuel Prices")
        plt.plot(pd.date_range(df.index[-1], periods=12, freq='M'), forecast, label="Forecasted Fuel Prices", linestyle='--')
        plt.title("Fuel Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Fuel Price (USD/barrel)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Interpretation
        forecast_mean = forecast.pct_change().mean()
        if forecast_mean > 0:
            print("Interpretation: The fuel price is forecasted to increase.")
            print("Recommendation: Prepare for rising fuel costs by exploring alternative energy sources or locking in fuel supply contracts at current prices.")
        elif forecast_mean < 0:
            print("Interpretation: The fuel price is forecasted to decrease.")
            print("Recommendation: Consider adjusting pricing models or capitalizing on lower fuel costs by expanding energy-intensive operations.")
        else:
            print("Interpretation: The fuel price is forecasted to remain stable.")
            print("Recommendation: Continue monitoring the market and explore opportunities for fuel cost optimization.")
    except Exception as e:
        logging.error(f"Error forecasting fuel prices: {e}")

# Main Function
def main():
    # Read data from CSV
    energy_data = read_energy_data("energy_data.csv")
    
    if energy_data is not None:
        # Energy trend analysis
        analyze_energy_trends(energy_data)
    
        # Analyze fundamental drivers
        analyze_fundamentals(energy_data)
    
        # Forecast energy demand
        forecast = forecast_energy_demand(energy_data)
    
        # Forecast fuel prices
        fuel_price_forecast = forecast_fuel_prices(energy_data)

if __name__ == "__main__":
    main()

