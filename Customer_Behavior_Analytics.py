import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # SMOTE for data balancing
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the dataset with error handling
def load_dataset(file_path):
    try:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)  # Load the dataset from the CSV file
            return data
        else:
            raise FileNotFoundError(f"File {file_path} not found.")  # Error if file is missing
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Load the dataset from the CSV file
customer_behavior_data = load_dataset('customer_data.csv')

# 2. Handling Missing Values with error handling
def handle_missing_values(data):
    try:
        # Define imputers for numerical and categorical data
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        # Impute numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = num_imputer.fit_transform(data[[col]]).ravel()  

        # Impute categorical columns
        for col in data.select_dtypes(include=[object]).columns:
            data[col] = cat_imputer.fit_transform(data[[col]]).ravel()  

        return data
    except Exception as e:
        print(f"Error handling missing values: {e}")
        return data

# Apply missing value handling
if customer_behavior_data is not None:
    customer_behavior_data = handle_missing_values(customer_behavior_data)

# 3. Sentiment Analysis for Customer Feedback Text
def sentiment_analysis_function(text_column):
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    try:
        return text_column.apply(analyze_sentiment)
    except Exception as e:
        print(f"Error performing sentiment analysis: {e}")
        return pd.Series([0]*len(text_column))

# 4. Behavioral Segmentation with KMeans Clustering 
def behavioral_segmentation_function(data, number_of_segments=5):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['Purchase_Frequency', 'Engagement_Score']])
        kmeans = KMeans(n_clusters=number_of_segments, random_state=42)
        customer_segments = kmeans.fit_predict(scaled_data)
        data['Customer_Segment'] = customer_segments
        return data
    except Exception as e:
        print(f"Error during behavioral segmentation: {e}")
        return data

# 5. Real-Time Dynamic Pricing Model
def dynamic_pricing_function(customer_data, market_conditions):
    try:
        customer_data['Dynamic_Price'] = customer_data['Purchase_Frequency'] * market_conditions['price_factor']
        return customer_data
    except Exception as e:
        print(f"Error calculating dynamic pricing: {e}")
        return customer_data

# 6. Data Balancing using SMOTE for Churn Prediction and Other Models
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

# 7. Proactive Retention Trigger for Churn Risk with Logistic Regression and Model Performance Metrics
def churn_prediction_function(customer_behavior):
    try:
        customer_behavior['Churn_Label'] = np.where(customer_behavior['Engagement_Score'] < 0.5, 1, 0)
        X = customer_behavior[['Purchase_Frequency', 'Engagement_Score']]
        y = customer_behavior['Churn_Label']

        # Data Balancing using SMOTE
        X_balanced, y_balanced = balance_data(X, y)

        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)

        # Logistic Regression with Hyperparameter Tuning
        logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear']
        }

        # KFold cross-validation
        min_class_size = min(np.bincount(y_balanced))
        cv_splits = KFold(n_splits=min(5, min_class_size), shuffle=True, random_state=42)

        grid_search = GridSearchCV(logistic_model, param_grid, cv=cv_splits)
        grid_search.fit(X_scaled, y_balanced)

        best_logistic_model = grid_search.best_estimator_

        # Predict churn
        X_test_scaled = scaler.transform(X)
        customer_behavior['Churn_Risk'] = best_logistic_model.predict(X_test_scaled)

        # Model performance metrics
        y_pred = best_logistic_model.predict(X_test_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print(f"Logistic Regression Performance Metrics:")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        return customer_behavior
    except Exception as e:
        print(f"Error predicting churn: {e}")
        return customer_behavior

# 8. Next-Best Action with Random Forest Classifier and Hyperparameter Tuning and Model Performance Metrics
def next_best_action_function(customer_behavior):
    try:
        X = customer_behavior[['Purchase_Frequency', 'Engagement_Score']]
        y = np.where(customer_behavior['Engagement_Score'] > 0.7, 'Recommend Product', 'Send Re-Engagement Offer')

        # Data Balancing using SMOTE
        X_balanced, y_balanced = balance_data(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)

        # Random Forest with Hyperparameter Tuning
        random_forest = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # KFold cross-validation
        min_class_size = min(np.bincount(np.array(y_balanced == 'Recommend Product').astype(int)))
        cv_splits = KFold(n_splits=min(5, min_class_size), shuffle=True, random_state=42)

        grid_search = GridSearchCV(random_forest, param_grid, cv=cv_splits)
        grid_search.fit(X_scaled, y_balanced)

        best_random_forest = grid_search.best_estimator_

        # Predict the next-best action
        X_test_scaled = scaler.transform(X)
        customer_behavior['Next_Best_Action'] = best_random_forest.predict(X_test_scaled)

        # Model performance metrics
        y_pred = best_random_forest.predict(X_test_scaled)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        print(f"Random Forest Performance Metrics:")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        return customer_behavior
    except Exception as e:
        print(f"Error predicting next best action: {e}")
        return customer_behavior

# 9. AI-Driven Business Interpreter 
def ai_business_interpreter(data):
    try:
        insights = []

        avg_sentiment = data['Sentiment_Score'].mean()
        median_sentiment = data['Sentiment_Score'].median()
        min_sentiment = data['Sentiment_Score'].min()
        max_sentiment = data['Sentiment_Score'].max()

        sentiment_category = "Neutral"
        if avg_sentiment > 0.1:
            sentiment_category = "Positive"
        elif avg_sentiment < -0.1:
            sentiment_category = "Negative"

        insights.append(f"Average customer sentiment is {sentiment_category} with an average score of {avg_sentiment:.2f}.")

        high_engaged_customers = data[(data['Purchase_Frequency'] > 5) & (data['Engagement_Score'] > 0.75)]
        low_engaged_customers = data[(data['Purchase_Frequency'] <= 5) & (data['Engagement_Score'] <= 0.75)]
        insights.append(f"Number of high engagement customers: {len(high_engaged_customers)}. Low engagement: {len(low_engaged_customers)}.")

        churn_risk_customers = data[data['Churn_Risk'] == 1]
        churn_rate = (len(churn_risk_customers) / len(data)) * 100
        insights.append(f"Churn Risk: {len(churn_risk_customers)} customers ({churn_rate:.2f}%).")

        price_sensitive_customers = data[data['Max_Price'] < data['Purchase_Amount']]
        if not price_sensitive_customers.empty:
            insights.append(f"{len(price_sensitive_customers)} price-sensitive customers.")

        platform_distribution = data['Platform'].value_counts()
        dominant_platform = platform_distribution.idxmax()
        insights.append(f"Most popular platform: {dominant_platform} with {platform_distribution[dominant_platform]} customers.")

        top_category = data['Preferred_Category'].value_counts().idxmax()
        insights.append(f"Top product category: {top_category}.")

        return insights
    except Exception as e:
        print(f"Error generating business insights: {e}")
        return []

# Apply sentiment analysis
if customer_behavior_data is not None:
    customer_behavior_data['Sentiment_Score'] = sentiment_analysis_function(customer_behavior_data['Feedback_Text'])

# Apply churn prediction with hyperparameter tuning and performance metrics
if customer_behavior_data is not None:
    customer_behavior_data = churn_prediction_function(customer_behavior_data)

# Apply next-best action with hyperparameter tuning and performance metrics
if customer_behavior_data is not None:
    customer_behavior_data = next_best_action_function(customer_behavior_data)

# Generate business insights
if customer_behavior_data is not None:
    insights = ai_business_interpreter(customer_behavior_data)
    for idx, insight in enumerate(insights, 1):
        print(f"AI Insight {idx}: {insight}")

# Visualizations
if customer_behavior_data is not None:
    # 1. Sentiment Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(customer_behavior_data['Sentiment_Score'], kde=True, bins=50, color='skyblue')
    plt.axvline(customer_behavior_data['Sentiment_Score'].mean(), color='r', linestyle='--', label=f'Average: {customer_behavior_data["Sentiment_Score"].mean():.2f}')
    plt.axvline(customer_behavior_data['Sentiment_Score'].median(), color='g', linestyle='-', label=f'Median: {customer_behavior_data["Sentiment_Score"].median():.2f}')
    plt.title("Sentiment Score Distribution with Key Metrics", fontsize=14)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Purchase Frequency vs Engagement Score by Platform
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Purchase_Frequency', y='Engagement_Score', hue='Platform', data=customer_behavior_data, s=100)
    plt.title("Purchase Frequency vs Engagement Score by Platform", fontsize=14)
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Engagement Score")
    plt.grid(True)
    plt.show()

    # 3. Purchase Amount Distribution by Platform
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Platform', y='Purchase_Amount', data=customer_behavior_data)
    plt.title("Purchase Amount Distribution by Platform", fontsize=14)
    plt.xlabel("Platform")
    plt.ylabel("Purchase Amount")
    plt.grid(True)
    plt.show()

    # 4. Max Price by Preferred Category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Preferred_Category', y='Max_Price', data=customer_behavior_data)
    plt.title("Max Price by Preferred Category", fontsize=14)
    plt.xlabel("Preferred Category")
    plt.ylabel("Max Price")
    plt.grid(True)
    plt.show()

    # Display the updated DataFrame with Next-Best Action and Churn Risk
    print("\nCustomer Behavior Data with Next-Best Action and Churn Risk:")
    print(customer_behavior_data[['Customer_ID', 'Engagement_Score', 'Churn_Risk', 'Next_Best_Action']])