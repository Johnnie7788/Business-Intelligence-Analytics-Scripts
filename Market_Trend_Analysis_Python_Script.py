import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data from the CSV file
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

# Step 2: Handle missing values automatically
def handle_missing_values(data):
    try:
        num_imputer = SimpleImputer(strategy='mean')
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = num_imputer.fit_transform(data[[col]])

        cat_imputer = SimpleImputer(strategy='most_frequent')
        for col in data.select_dtypes(include=[object]).columns:
            data[col] = cat_imputer.fit_transform(data[[col]])

        return data
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return data

# Step 3: Handle class imbalance using SMOTE
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

# Step 4: Dynamic Market Share Adjustment and Ad Spend Efficiency Index (ASEI)
def calculate_dynamic_metrics(data):
    try:
        data['ASEI'] = data['Sales_Volume'] / data['Ad_Spend']
        data['Adjusted_Market_Share'] = data['Market_Share'] + np.random.uniform(-0.02, 0.02, len(data))
        return data
    except Exception as e:
        print(f"Error in calculating dynamic metrics: {e}")
        return data

# Step 5: AI-Powered Sentiment-Driven Sales Prediction using Random Forest with Hyperparameter Tuning
def ai_sales_forecast(data):
    try:
        X = data[['Customer_Sentiment', 'Ad_Spend', 'ASEI']]
        y = data['Sales_Volume']

        X_balanced, y_balanced = balance_data(X, y)

        rf = RandomForestRegressor(random_state=42)

        # Hyperparameter Tuning using GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Cross-validation setup
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=kfold, n_jobs=-1)

        # Fitting the model with best parameters
        grid_search.fit(X_balanced, y_balanced)
        best_rf_model = grid_search.best_estimator_

        # Predict future sales based on customer sentiment
        data['Predicted_Sales_RF'] = best_rf_model.predict(X)

        return data
    except Exception as e:
        print(f"Error in sales forecasting: {e}")
        return data

# Step 6: Multi-Dimensional Time Series Clustering with Hyperparameter Tuning for KMeans
def time_series_clustering(data, min_clusters=2, max_clusters=10):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['Sales_Volume', 'Ad_Spend', 'Customer_Sentiment']])

        best_k = min_clusters
        best_inertia = np.inf

        # Find the optimal number of clusters using a range search
        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertia = kmeans.inertia_
            if inertia < best_inertia:
                best_inertia = inertia
                best_k = k

        # Fit the best KMeans model
        best_kmeans = KMeans(n_clusters=best_k, random_state=42)
        data['Trend_Cluster'] = best_kmeans.fit_predict(scaled_data)

        return data
    except Exception as e:
        print(f"Error in time series clustering: {e}")
        return data

# Step 7: Social Media Sentiment Tracking Simulation
def social_media_sentiment_tracking(data):
    try:
        data['Social_Sentiment'] = np.random.uniform(-1, 1, len(data))
        data['Sentiment_Alert'] = np.where(data['Social_Sentiment'] < -0.5, 'Negative Spike', 'Normal')
        return data
    except Exception as e:
        print(f"Error in tracking social media sentiment: {e}")
        return data

# Step 8: Dynamic Marketing Recommendation System
def recommendation_system(data):
    try:
        recommendations = []
        for index, row in data.iterrows():
            if row['ASEI'] > data['ASEI'].mean():
                recommendation = f"Increase ad spend for {row['Product_Category']} (Cluster {row['Trend_Cluster']})"
            else:
                recommendation = f"Reduce ad spend or optimize strategy for {row['Product_Category']} (Cluster {row['Trend_Cluster']})"
            
            recommendations.append(recommendation)
        
        data['Marketing_Recommendation'] = recommendations
        
        return data
    except Exception as e:
        print(f"Error in generating recommendations: {e}")
        return data

# Step 9: Automatic AI Interpretation of Visualizations
def ai_interpreter(data):
    try:
        insights = []

        if data['ASEI'].mean() > 1.0:
            insights.append("Ad spend is being efficiently converted into sales.")
        else:
            insights.append("Ad spend may need optimization.")

        if len(data[data['Social_Sentiment'] < -0.5]) > 0:
            insights.append("Negative sentiment spike detected, consider a PR intervention.")

        return insights
    except Exception as e:
        print(f"Error in AI interpretation: {e}")
        return []

# Step 10: Automatic Report Generation
def generate_report(data):
    try:
        report = f"Market Trend Report:\n"
        report += f"Average ASEI: {data['ASEI'].mean():.2f}\n"
        report += f"Total Products with Negative Social Sentiment: {len(data[data['Social_Sentiment'] < 0])}\n"
        report += f"Top Recommendations:\n"

        for rec in data['Marketing_Recommendation'].unique():
            report += f"- {rec}\n"

        return report
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error in generating report."

# Step 11: Visualize the Market Trends with enhanced visuals
def visualize_trends(data):
    try:
        plt.figure(figsize=(15, 10))

        # Visualization 1: Sales Volume vs Predicted Sales by Random Forest
        plt.subplot(2, 2, 1)
        sns.regplot(x='Predicted_Sales_RF', y='Sales_Volume', data=data, scatter_kws={'s':100}, line_kws={'color':'red'})
        plt.title('Actual vs Predicted Sales (Random Forest)', fontsize=14)
        plt.xlabel('Predicted Sales (RF)')
        plt.ylabel('Actual Sales Volume')
        plt.grid(True)

        # Visualization 2: Dynamic Market Share Over Time
        plt.subplot(2, 2, 2)
        sns.lineplot(x='Date', y='Adjusted_Market_Share', hue='Product_Category', data=data, marker='o')
        plt.title('Dynamic Market Share Over Time', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Adjusted Market Share')
        plt.grid(True)

        # Visualization 3: Cluster Heatmap for Hidden Patterns
        plt.subplot(2, 2, 3)
        cluster_data = data[['Sales_Volume', 'Customer_Sentiment', 'Ad_Spend', 'Trend_Cluster']].pivot_table(index='Trend_Cluster')
        sns.heatmap(cluster_data, cmap='coolwarm', annot=True, fmt='.2f')
        plt.title('Cluster Analysis Heatmap', fontsize=14)
        plt.xlabel('Features')
        plt.ylabel('Cluster')

        # Visualization 4: Sales Volume vs ASEI (Ad Spend Efficiency Index)
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='ASEI', y='Sales_Volume', hue='Trend_Cluster', size='Sales_Volume', sizes=(50, 200), data=data)
        plt.title('Sales Volume vs ASEI (with Clusters)', fontsize=14)
        plt.xlabel('Ad Spend Efficiency Index (ASEI)')
        plt.ylabel('Sales Volume')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in visualization: {e}")

# Full workflow
file_path = 'market_trend_data.csv'
market_trend_data = load_data(file_path)

if market_trend_data is not None:
    # Handle missing values
    market_trend_data = handle_missing_values(market_trend_data)

    # Apply dynamic metrics
    market_trend_data = calculate_dynamic_metrics(market_trend_data)

    # Apply AI-Powered Sales Forecasting
    market_trend_data = ai_sales_forecast(market_trend_data)

    # Apply time series clustering
    market_trend_data = time_series_clustering(market_trend_data)

    # Apply social media sentiment tracking
    market_trend_data = social_media_sentiment_tracking(market_trend_data)

    # Generate marketing recommendations
    market_trend_data = recommendation_system(market_trend_data)

    # Automatically generate insights and report
    ai_insights = ai_interpreter(market_trend_data)
    report = generate_report(market_trend_data)
    
    # Display AI insights
    for idx, insight in enumerate(ai_insights, 1):
        print(f"AI Insight {idx}: {insight}")
    
    # Display report
    print(report)
    
    # Visualize market trends
    visualize_trends(market_trend_data)

