#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class B2BCustomerInsights:
    def __init__(self, sales_data_path, marketing_data_path, crm_data_path):
        """Initialize the class with paths to sales, marketing, and CRM data."""
        self.sales_data_path = sales_data_path
        self.marketing_data_path = marketing_data_path
        self.crm_data_path = crm_data_path
        self.combined_data = None

    def load_data(self):
        """Load sales, marketing, and CRM data."""
        try:
            logging.info("Loading datasets...")
            self.sales_data = pd.read_csv(self.sales_data_path)
            self.marketing_data = pd.read_csv(self.marketing_data_path)
            self.crm_data = pd.read_csv(self.crm_data_path)
            logging.info("Datasets loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise

    def aggregate_data(self):
        """Aggregate sales, marketing, and CRM data into a comprehensive customer profile."""
        try:
            logging.info("Aggregating data...")
            combined = pd.merge(self.sales_data, self.marketing_data, on="customer_id", how="outer")
            self.combined_data = pd.merge(combined, self.crm_data, on="customer_id", how="outer")
            self.combined_data.fillna(0, inplace=True)
            logging.info("Data aggregation completed successfully.")
        except Exception as e:
            logging.error(f"Error during data aggregation: {e}")
            raise

    def generate_insights(self):
        """Identify patterns and trends specific to B2B customers."""
        try:
            logging.info("Generating insights...")
            numeric_columns = self.combined_data.select_dtypes(include=np.number).columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.combined_data[numeric_columns])

            kmeans = KMeans(n_clusters=5, random_state=42)
            self.combined_data["customer_segment"] = kmeans.fit_predict(scaled_data)

            plt.figure(figsize=(10, 6))
            sns.countplot(x="customer_segment", data=self.combined_data, palette="viridis")
            plt.title("Customer Segments Distribution")
            plt.xlabel("Segment")
            plt.ylabel("Number of Customers")
            plt.show()

            logging.info("Insights generation completed successfully.")
        except Exception as e:
            logging.error(f"Error during insights generation: {e}")
            raise

    def hyperparameter_tuning(self, max_clusters=10):
        """Determine the optimal number of clusters using the Elbow Method."""
        try:
            logging.info("Determining the optimal number of clusters...")
            numeric_columns = self.combined_data.select_dtypes(include=np.number).columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.combined_data[numeric_columns])

            distortions = []
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                distortions.append(kmeans.inertia_)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_clusters + 1), distortions, marker='o')
            plt.title("Elbow Method for Optimal Clusters")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Distortion")
            plt.show()

            logging.info("Optimal clusters analysis completed successfully.")
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            raise

    def recommend_engagement_strategies(self):
        """Provide automatic recommendations for improving customer engagement and satisfaction."""
        try:
            logging.info("Generating recommendations...")
            recommendations = []

            for _, row in self.combined_data.iterrows():
                if row["customer_segment"] == 0:
                    recommendations.append("Offer loyalty discounts")
                elif row["customer_segment"] == 1:
                    recommendations.append("Upsell premium products")
                elif row["customer_segment"] == 2:
                    recommendations.append("Provide targeted marketing campaigns")
                elif row["customer_segment"] == 3:
                    recommendations.append("Focus on customer support")
                else:
                    recommendations.append("Engage with personalized content")

            self.combined_data["recommendations"] = recommendations
            logging.info("Recommendations generated successfully.")
        except Exception as e:
            logging.error(f"Error during recommendation generation: {e}")
            raise

    def visualize_recommendations(self):
        """Visualize the distribution of recommendations."""
        try:
            logging.info("Visualizing recommendations...")
            plt.figure(figsize=(10, 6))
            sns.countplot(y="recommendations", data=self.combined_data, palette="coolwarm", order=self.combined_data["recommendations"].value_counts().index)
            plt.title("Recommendation Distribution")
            plt.xlabel("Count")
            plt.ylabel("Recommendation")
            plt.show()
            logging.info("Visualization completed successfully.")
        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise

    def save_results(self, output_path):
        """Save the combined data with insights and recommendations to a CSV file."""
        try:
            logging.info("Saving results...")
            self.combined_data.to_csv(output_path, index=False)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise

if __name__ == "__main__":
    # Define file paths
    sales_path = "sales_data.csv"
    marketing_path = "marketing_data.csv"
    crm_path = "crm_data.csv"

    # Initialize the insights class
    insights = B2BCustomerInsights(sales_path, marketing_path, crm_path)

    try:
        # Load and process data
        insights.load_data()
        insights.aggregate_data()

        # Determine optimal clusters
        insights.hyperparameter_tuning(max_clusters=10)

        # Generate insights and recommendations
        insights.generate_insights()
        insights.recommend_engagement_strategies()

        # Visualize and save results
        insights.visualize_recommendations()
        insights.save_results("b2b_customer_insights.csv")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")

