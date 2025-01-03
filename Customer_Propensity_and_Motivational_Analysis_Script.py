#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging

nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomerAnalysis:
    def __init__(self, data_path):
        """Initialize the CustomerAnalysis class with the data path."""
        self.data_path = data_path
        self.data = None
        self.model = None

    def load_data(self):
        """Load the dataset from the specified path."""
        try:
            logging.info("Loading dataset...")
            self.data = pd.read_csv(self.data_path)
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")

    def preprocess_data(self):
        """Preprocess data for modeling."""
        try:
            logging.info("Preprocessing data...")
            # Handle missing values
            self.data.fillna({"age": self.data["age"].mean(), "income": self.data["income"].median()}, inplace=True)
            
            # Encode categorical variables
            self.data = pd.get_dummies(self.data, drop_first=True)
            logging.info("Data preprocessing completed.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")

    def predictive_modeling(self):
        """Perform predictive modeling using Random Forest and XGBoost."""
        try:
            logging.info("Running predictive modeling...")
            
            # Define features and target
            X = self.data.drop("purchase", axis=1)
            y = self.data["purchase"]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Random Forest
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            logging.info("Random Forest Classification Report:\n" + classification_report(y_test, rf_preds))
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            logging.info("XGBoost Classification Report:\n" + classification_report(y_test, xgb_preds))

            self.model = rf_model  # Use Random Forest as the default model
        except Exception as e:
            logging.error(f"Error during predictive modeling: {e}")

    def sentiment_analysis(self, text_column):
        """Perform sentiment analysis on customer reviews or feedback."""
        try:
            logging.info("Performing sentiment analysis...")
            sia = SentimentIntensityAnalyzer()

            self.data['sentiment'] = self.data[text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
            
            # Plot sentiment distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['sentiment'], kde=True, bins=20, color='blue')
            plt.title('Customer Sentiment Distribution')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            plt.show()
            logging.info("Sentiment analysis completed successfully.")
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")

    def motivational_trend_analysis(self, text_column):
        """Perform motivational trend analysis using word clouds and topic modeling."""
        try:
            logging.info("Analyzing customer motivations...")

            # Generate word cloud
            text_data = ' '.join(self.data[text_column].dropna())
            wordcloud = WordCloud(background_color='white', max_words=100).generate(text_data)

            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Customer Motivational Word Cloud')
            plt.show()

            # Topic modeling using LDA
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            text_matrix = vectorizer.fit_transform(self.data[text_column].dropna())

            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(text_matrix)

            logging.info("Top words per topic:")
            for idx, topic in enumerate(lda.components_):
                logging.info(f"Topic {idx + 1}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        except Exception as e:
            logging.error(f"Error during motivational trend analysis: {e}")

    def visualize_feature_importance(self):
        """Visualize feature importance from the Random Forest model."""
        try:
            logging.info("Visualizing feature importance...")
            feature_importances = self.model.feature_importances_
            features = self.data.drop("purchase", axis=1).columns

            sorted_idx = np.argsort(feature_importances)

            plt.figure(figsize=(10, 6))
            plt.barh(features[sorted_idx], feature_importances[sorted_idx], color='teal')
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance in Predicting Customer Purchase')
            plt.show()
            logging.info("Feature importance visualization completed successfully.")
        except Exception as e:
            logging.error(f"Error during feature importance visualization: {e}")

if __name__ == "__main__":
    analysis = CustomerAnalysis(data_path="customer_data.csv")

    # Load data
    analysis.load_data()

    # Preprocess data
    analysis.preprocess_data()

    # Predictive modeling
    analysis.predictive_modeling()

    # Sentiment analysis
    analysis.sentiment_analysis(text_column="customer_reviews")

    # Motivational trend analysis
    analysis.motivational_trend_analysis(text_column="customer_reviews")

    # Feature importance visualization
    analysis.visualize_feature_importance()

