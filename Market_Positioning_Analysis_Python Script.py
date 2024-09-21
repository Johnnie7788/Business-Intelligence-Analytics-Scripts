#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px
from prophet import Prophet

# 1. Reading Data from CSV
def read_data(csv_file):
    """
    Reads the market positioning data from the CSV file.
    """
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    print("Data successfully read from CSV.")
    return df

# 2. Interactive Market Share Analysis with Plotly
def interactive_market_share(df):
    """
    Analyzes market share trends using an interactive Plotly visualization.
    Automatically interprets the trend and makes recommendations.
    """
    fig = px.line(df, x='Date', y='Market_Share (%)', color='Company', title='Interactive Market Share Trends')
    fig.update_layout(hovermode='x unified')
    fig.show()

    # Interpretation
    avg_market_share = df.groupby('Company')['Market_Share (%)'].mean()
    leading_company = avg_market_share.idxmax()
    trailing_company = avg_market_share.idxmin()
    
    print(f"Interpretation: {leading_company} has the highest average market share at {avg_market_share.max():.2f}%.")
    print(f"Recommendation: {leading_company} should focus on maintaining its leadership position through innovation and customer retention strategies.")
    print(f"{trailing_company} has the lowest market share at {avg_market_share.min():.2f}%.")
    print(f"Recommendation: {trailing_company} should consider increasing marketing spend or exploring new market segments to boost market share.")
    
    return avg_market_share  # Return for final analysis

# 3. Predictive Forecasting with Prophet for Market Share
def forecast_market_share(df):
    """
    Forecasts future market share for each company using Prophet.
    """
    for company in df['Company'].unique():
        company_data = df[df['Company'] == company][['Date', 'Market_Share (%)']]
        company_data = company_data.rename(columns={'Date': 'ds', 'Market_Share (%)': 'y'})
        
        model = Prophet()
        model.fit(company_data)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        
        fig = model.plot(forecast)
        plt.title(f"Market Share Forecast for {company}")
        plt.show()

# 4. Marketing Spend vs Revenue Analysis
def marketing_vs_revenue_analysis(df):
    """
    Analyzes the relationship between marketing spend and revenue.
    Automatically interprets the result and provides recommendations.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Marketing_Spend (USD)', y='Revenue (USD)', hue='Company', data=df, s=100)
    plt.title("Marketing Spend vs Revenue by Company")
    plt.xlabel("Marketing Spend (USD)")
    plt.ylabel("Revenue (USD)")
    plt.tight_layout()
    plt.show()

    # Interpretation using Linear Regression
    X = df[['Marketing_Spend (USD)']]
    y = df['Revenue (USD)']
    model = LinearRegression().fit(X, y)
    print(f"Linear Regression Coefficient (Marketing Spend impact on Revenue): {model.coef_[0]:.2f}")
    
    if model.coef_[0] > 0:
        print("Interpretation: Increased marketing spend is positively correlated with revenue.")
        print("Recommendation: Companies should consider optimizing their marketing budgets to further increase revenue.")
    else:
        print("Interpretation: Marketing spend has a minimal or negative impact on revenue.")
        print("Recommendation: Companies should reassess their marketing strategies to ensure more effective use of their marketing budgets.")
    
    return model.coef_[0]  # Return for final analysis

# 5. Customer Sentiment Analysis
def customer_sentiment_analysis(df):
    """
    Analyzes customer sentiment scores across companies.
    Automatically interprets the sentiment and gives recommendations.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Company', y='Customer_Sentiment', data=df)
    plt.title("Customer Sentiment Scores by Company")
    plt.ylabel("Customer Sentiment (1 to 5)")
    plt.tight_layout()
    plt.show()

    # Interpretation
    avg_sentiment = df.groupby('Company')['Customer_Sentiment'].mean()
    best_sentiment_company = avg_sentiment.idxmax()
    worst_sentiment_company = avg_sentiment.idxmin()

    print(f"Interpretation: {best_sentiment_company} has the highest average customer sentiment score at {avg_sentiment.max():.2f}.")
    print(f"Recommendation: {best_sentiment_company} should continue to invest in customer satisfaction strategies to maintain its positive brand image.")
    print(f"{worst_sentiment_company} has the lowest customer sentiment score at {avg_sentiment.min():.2f}.")
    print(f"Recommendation: {worst_sentiment_company} should investigate customer pain points and improve service or product quality to enhance customer perception.")
    
    return avg_sentiment  # Return for final analysis

# 6. Present Company Position Based on Results
def present_position(avg_market_share, coef, avg_sentiment):
    """
    Summarizes the present position of each company based on market share, marketing spend efficiency, and customer sentiment.
    """
    print("\n--- Present Company Positions ---")
    
    for company in avg_market_share.index:
        share_position = avg_market_share[company]
        sentiment_position = avg_sentiment[company]

        # Determine position based on average market share, sentiment, and marketing efficiency
        if share_position >= 30 and sentiment_position >= 4 and coef > 0:
            position = "Leader"
        elif share_position >= 20 and sentiment_position >= 3:
            position = "Stable"
        else:
            position = "Needs Improvement"
        
        print(f"Company: {company} is currently positioned as a '{position}' in the market.")
        
# Main function to run all analyses
def main():
    csv_file = "market_positioning_data.csv"  
    df = read_data(csv_file)

    avg_market_share = interactive_market_share(df)
    forecast_market_share(df)
    coef = marketing_vs_revenue_analysis(df)
    avg_sentiment = customer_sentiment_analysis(df)
    present_position(avg_market_share, coef, avg_sentiment)

if __name__ == "__main__":
    main()

