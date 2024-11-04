#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Reading the dataset
df = pd.read_csv('AI-Driven Sentiment Prediction and Customer Insights Platform.csv')

# Preprocess the 'Review_Text' column
df['Processed_Review_Text'] = df['Review_Text'].apply(preprocess_text)

# Perform sentiment analysis on the processed text
df['Predicted_Sentiment'] = df['Processed_Review_Text'].apply(analyze_sentiment)

# Machine Learning Model: Predicting the impact of sentiment on customer ratings using Support Vector Machine (SVM)
def predict_review_ratings(df):
    # Create Sentiment Score
    df['Sentiment_Score'] = df['Processed_Review_Text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

    # Prepare features and target
    X = df[['Sentiment_Score']]
    y = df['Review_Rating']  # Using Review_Rating as a proxy for brand reputation

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM model setup with cross-validation using GridSearchCV
    svr = SVR()

    # Grid Search for hyperparameter tuning
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100], 'epsilon': [0.1, 0.2, 0.5]}
    grid_search = GridSearchCV(svr, parameters, cv=5, scoring='r2')

    # Fitting the model using cross-validation
    grid_search.fit(X_train, y_train)

    # Best estimator from GridSearch
    best_model = grid_search.best_estimator_

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Predict on full data for analysis purposes
    df['Predicted_Review_Rating'] = best_model.predict(X)

    # Evaluate the model using R², MAE, MSE
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Support Vector Machine Model Evaluation:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    return best_model, df

# Time Series Forecasting: Predict future sentiment trends using ARIMA
def forecast_sentiment_trends(df):
    # Convert sentiment to numerical values for time series
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment_Numeric'] = df['Predicted_Sentiment'].map(sentiment_mapping)

    # Prepare time series data (assuming 'Date' column exists in the dataset)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the date column is in datetime format
    df.set_index('Date', inplace=True)

    # Fit ARIMA model
    model = sm.tsa.ARIMA(df['Sentiment_Numeric'], order=(1, 1, 1))  # Adjust order as necessary
    model_fit = model.fit()

    # Forecast future sentiment trends
    forecast = model_fit.forecast(steps=30)  # Forecast for the next 30 days

    return forecast

# Interactive Data Visualizations using Plotly
def visualize_sentiment_trends(df):
    # Visualize historical sentiment trends
    fig = px.line(df, x=df.index, y='Sentiment_Numeric', title='Historical Sentiment Trends')

    # Show visualization
    fig.show()

    # Visualization of sentiment distribution
    fig_pie = px.pie(df, names='Predicted_Sentiment', title='Sentiment Distribution', hole=0.3)
    fig_pie.show()

# Enhanced automatic interpreter and recommendation
def interpret_results(df):
    sentiment_counts = df['Predicted_Sentiment'].value_counts()
    total_reviews = len(df)
    pos_percentage = (sentiment_counts.get('Positive', 0) / total_reviews) * 100
    neg_percentage = (sentiment_counts.get('Negative', 0) / total_reviews) * 100
    neu_percentage = (sentiment_counts.get('Neutral', 0) / total_reviews) * 100

    # Detailed Interpretation for managerial action
    interpretation = f"""
    Total Reviews Analyzed: {total_reviews}
    Positive Reviews: {pos_percentage:.2f}% ({sentiment_counts.get('Positive', 0)} reviews)
    Negative Reviews: {neg_percentage:.2f}% ({sentiment_counts.get('Negative', 0)} reviews)
    Neutral Reviews: {neu_percentage:.2f}% ({sentiment_counts.get('Neutral', 0)} reviews)

    Detailed Insights and Managerial Recommendations:

    **1. Positive Sentiment ({pos_percentage:.2f}%):**
    - Customers are largely satisfied, indicating the product or service is generally well-received. 
    - **Managerial Action:**
        - **Upsell Opportunities:** Leverage this positivity by offering loyalty programs or exclusive promotions. This can further drive customer engagement and lifetime value.
        - **Brand Ambassadors:** Engage positive reviewers as brand ambassadors or for customer testimonials to reinforce trust and attract new customers.
        - **Referral Programs:** Consider launching referral campaigns, encouraging satisfied customers to bring new leads through word-of-mouth.

    **2. Negative Sentiment ({neg_percentage:.2f}%):**
    - The negative feedback indicates areas that require immediate attention to mitigate customer dissatisfaction.
    - **Managerial Action:**
        - **Root Cause Analysis:** Deep dive into negative reviews to identify the most frequent complaints. Categorize the issues (e.g., product quality, service response time, delivery delays) and prioritize corrective actions.
        - **Customer Recovery Strategy:** Implement a customer recovery strategy, offering proactive solutions such as refunds, discounts, or replacement products for affected customers. Communicate these efforts to regain trust.
        - **Process Improvement:** Consider addressing operational inefficiencies or quality control issues that contribute to negative reviews.
        - **Long-Term Improvement:** Ensure ongoing monitoring of negative reviews and implement systematic improvements in those areas over time to reduce future complaints.

    **3. Neutral Sentiment ({neu_percentage:.2f}%):**
    - Neutral reviews reflect indifference, which can indicate missed opportunities for deeper customer engagement.
    - **Managerial Action:**
        - **Customer Engagement:** Reach out to neutral reviewers through follow-up surveys to gather additional feedback. This will help to better understand their reservations and convert them into satisfied or loyal customers.
        - **Product Refinement:** Analyze the content of neutral reviews to determine if there are areas where expectations were met but not exceeded. Product enhancements based on neutral feedback can significantly boost overall satisfaction.
        - **Enhancing Customer Experience:** Neutral reviews may also suggest areas where personalization or a more tailored customer experience can improve satisfaction. Experiment with personalized offerings or more targeted communications.

    **Overall Strategic Recommendations:**
    - **Focus on Customer-Centric Initiatives:** High positive sentiment is a strong indicator of brand loyalty. Invest in nurturing these customers while addressing issues faced by those with negative or neutral experiences. Consider hosting online feedback sessions or using social listening tools to stay proactive in managing customer sentiment.
    - **Data-Driven Marketing:** Utilize sentiment insights in your marketing strategies. For example, create campaigns showcasing positive reviews to enhance brand reputation and promote features addressing common negative feedback points.
    - **Ongoing Monitoring:** Continue sentiment analysis on a regular basis to track shifts in customer opinion. This will enable management to make dynamic, data-driven decisions about product development, customer service, and overall business strategy.
    """
    return interpretation

# Preprocess and analyze sentiment
df['Processed_Review_Text'] = df['Review_Text'].apply(preprocess_text)
df['Predicted_Sentiment'] = df['Processed_Review_Text'].apply(analyze_sentiment)

# Run prediction model using 'Review_Rating' as target
model, df = predict_review_ratings(df)

# Run sentiment trend forecasting
forecast = forecast_sentiment_trends(df)

# Visualize sentiment trends
visualize_sentiment_trends(df)

# Run interpretation and print detailed recommendations
interpretation = interpret_results(df)
print(interpretation)

# Save the processed dataset with results
df.to_csv('AI-Driven Sentiment Prediction and Customer Insights Platform.csv', index=False)

"""
If you're dealing with big datasets, the above approach can be scaled using PySpark for distributed processing. Here's how you would modify the code to use PySpark:

1. Initialize a Spark session:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("AI_Driven_Sentiment_Analysis").getOrCreate()

2. Read large datasets:
    df_spark = spark.read.csv('AI-Driven_Customer_Sentiment_Analysis.csv', header=True, inferSchema=True)

3. Use PySpark's ML library for sentiment scoring, preprocessing, and predictions:
    # Example for sentiment scoring using PySpark UDFs:
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType, DoubleType

    sentiment_udf = udf(lambda text: SentimentIntensityAnalyzer().polarity_scores(text)['compound'], DoubleType())
    df_spark = df_spark.withColumn("Sentiment_Score", sentiment_udf(col("Processed_Review_Text")))

4. PySpark's MLlib could be used for building scalable models.
"""

