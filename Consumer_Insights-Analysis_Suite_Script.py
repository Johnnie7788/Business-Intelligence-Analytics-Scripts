#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import logging

# Statistical libraries
from scipy import stats
from statsmodels.formula.api import ols

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# NLP for Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Database connection (assuming use of SQL via DataBricks or similar)
import sqlalchemy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(survey_filepath, sql_connection_string):
    """
    Load and preprocess survey data from CSV and transactional data from SQL database.
    """
    try:
        # Load survey data
        survey_data = pd.read_csv(survey_filepath)
        logging.info(f"Survey data loaded with {survey_data.shape[0]} records.")
    except Exception as e:
        logging.error(f"Error loading survey data: {e}")
        return None
    
    try:
        # Load transactional data from SQL database
        engine = sqlalchemy.create_engine(sql_connection_string)
        transactional_data = pd.read_sql('SELECT * FROM transactions', engine)
        logging.info(f"Transactional data loaded with {transactional_data.shape[0]} records.")
    except Exception as e:
        logging.error(f"Error loading transactional data from SQL: {e}")
        return None

    # Merge datasets on customer ID
    try:
        data = pd.merge(survey_data, transactional_data, on='customer_id', how='inner')
        logging.info(f"Data merged with {data.shape[0]} records.")
    except KeyError as e:
        logging.error(f"Merge failed: {e}")
        return None
    
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    logging.info("Missing values handled.")
    
    return data

def sentiment_analysis(data, text_column):
    """
    Perform sentiment analysis on survey open-ended responses.
    """
    if text_column not in data.columns:
        logging.error(f"{text_column} column not found in data.")
        return data
    
    logging.info("Starting sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()
    data[text_column] = data[text_column].fillna("")  # Handle missing text entries
    data['sentiment_score'] = data[text_column].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    logging.info("Sentiment analysis completed.")
    return data

def customer_segmentation(data):
    """
    Segment customers using KMeans clustering based on behavioral data.
    """
    required_columns = ['purchase_frequency', 'average_order_value', 'sentiment_score']
    if not all(column in data.columns for column in required_columns):
        missing_cols = set(required_columns) - set(data.columns)
        logging.error(f"Missing columns for segmentation: {missing_cols}")
        return data

    logging.info("Starting customer segmentation...")
    features = data[required_columns]
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['segment'] = kmeans.fit_predict(features)
    silhouette_avg = silhouette_score(features, data['segment'])
    logging.info(f"Customer segmentation completed with silhouette score: {silhouette_avg:.2f}")
    return data

def inferential_statistics(data):
    """
    Perform inferential statistics to understand the impact of sentiment on purchase behavior.
    """
    if 'total_spent' not in data.columns or 'sentiment_score' not in data.columns:
        logging.error("Required columns for inferential statistics are missing.")
        return None
    
    logging.info("Performing inferential statistics...")
    model = ols('total_spent ~ sentiment_score', data=data).fit()
    logging.info(f"Regression analysis completed with R-squared: {model.rsquared:.2f}")
    return model.summary()

def quasi_experiment_analysis(data):
    """
    Analyze the impact of a marketing campaign using a quasi-experimental design.
    """
    if 'date' not in data.columns or 'total_spent' not in data.columns:
        logging.error("Required columns for quasi-experimental analysis are missing.")
        return None
    
    logging.info("Starting quasi-experimental analysis...")
    pre_campaign = data[data['date'] < '2023-01-01']
    post_campaign = data[data['date'] >= '2023-01-01']
    
    pre_avg = pre_campaign['total_spent'].mean()
    post_avg = post_campaign['total_spent'].mean()
    
    t_stat, p_value = stats.ttest_ind(post_campaign['total_spent'], pre_campaign['total_spent'])
    logging.info(f"Quasi-experiment analysis completed with p-value: {p_value:.4f}")
    
    return {'pre_avg': pre_avg, 'post_avg': post_avg, 't_stat': t_stat, 'p_value': p_value}

def predictive_modeling(data):
    """
    Build a predictive model to forecast future sales based on historical data.
    """
    required_columns = ['sentiment_score', 'purchase_frequency', 'average_order_value', 'total_spent']
    if not all(column in data.columns for column in required_columns):
        missing_cols = set(required_columns) - set(data.columns)
        logging.error(f"Missing columns for predictive modeling: {missing_cols}")
        return None
    
    logging.info("Starting predictive modeling...")
    X = data[['sentiment_score', 'purchase_frequency', 'average_order_value']]
    y = data['total_spent']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    logging.info(f"Predictive model completed with R-squared: {score:.2f}")
    
    return reg

def generate_reports(data):
    """
    Generate visual reports for stakeholders.
    """
    logging.info("Generating reports...")
    
    # Sales over time by customer segment
    if 'date' in data.columns and 'total_spent' in data.columns and 'segment' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x='date', y='total_spent', hue='segment')
        plt.title('Sales Over Time by Customer Segment')
        plt.savefig('sales_over_time.png')
        plt.close()
        logging.info("Sales over time report generated.")
    else:
        logging.error("Required columns for sales over time report are missing.")
    
    # Sentiment distribution
    if 'sentiment_score' in data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data['sentiment_score'], bins=20, kde=True)
        plt.title('Customer Sentiment Distribution')
        plt.savefig('sentiment_distribution.png')
        plt.close()
        logging.info("Sentiment distribution report generated.")
    
    # Average Order Value by Segment
    if 'segment' in data.columns and 'average_order_value' in data.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=data, x='segment', y='average_order_value')
        plt.title('Average Order Value by Customer Segment')
        plt.savefig('aov_by_segment.png')
        plt.close()
        logging.info("Average Order Value report generated.")
    
    logging.info("All reports generated.")

def main(survey_filepath, sql_connection_string):
    # Load and preprocess data
    data = load_data(survey_filepath, sql_connection_string)
    if data is None:
        logging.error("Data loading failed. Exiting...")
        return
    
    # Sentiment Analysis
    data = sentiment_analysis(data, 'open_ended_response')
    
    # Customer Segmentation
    data = customer_segmentation(data)
    
    # Inferential Statistics
    regression_summary = inferential_statistics(data)
    if regression_summary:
        print(regression_summary)
    
    # Quasi-Experimental Analysis
    quasi_results = quasi_experiment_analysis(data)
    if quasi_results:
        print(f"Pre-Campaign Average Spend: {quasi_results['pre_avg']}")
        print(f"Post-Campaign Average Spend: {quasi_results['post_avg']}")
        print(f"T-Statistic: {quasi_results['t_stat']}")
        print(f"P-Value: {quasi_results['p_value']}")
    
    # Predictive Modeling
    model = predictive_modeling(data)
    
    # Generate Reports
    generate_reports(data)
    
    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Research Analysis Script")
    parser.add_argument('--survey_filepath', type=str, required=True, help="Path to the survey CSV file")
    parser.add_argument('--sql_connection_string', type=str, required=True, help="SQL connection string for transactional data")
    args = parser.parse_args()
    
    main(args.survey_filepath, args.sql_connection_string)

