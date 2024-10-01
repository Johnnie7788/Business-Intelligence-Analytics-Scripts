
# AI-Driven Sentiment Prediction and Customer Insights Platform

## Overview
The AI-Driven Sentiment Prediction and Customer Insights Platform script is specifically designed for businesses to analyze customer feedback, predict sentiment trends, and derive actionable insights. It leverages a machine learning model, sentiment analysis tools, and forecasting techniques to provide a holistic view of customer sentiment and its impact on business performance.

### Key Features:
1. **Sentiment Prediction**: Uses AI-driven sentiment analysis to categorize customer feedback as positive, negative, or neutral.
2. **Customer Ratings Prediction**: Predicts customer ratings based on sentiment scores using Support Vector Machines (SVM) with hyperparameter tuning.
3. **Time Series Forecasting**: Forecasts future sentiment trends using ARIMA, alerting management to potential shifts in customer opinion.
4. **Data Visualization**: Interactive visualizations using Plotly for dynamic exploration of sentiment trends and distributions.
5. **AI-Driven Business Insights**: Automatically generates managerial recommendations based on sentiment analysis, helping businesses enhance customer satisfaction and loyalty.

This platform is ideal for businesses seeking to leverage AI-powered analytics to understand customer sentiment, predict future trends, and optimize decision-making.

## Key Features & Functionality

### 1. Data Loading and Preprocessing
The platform loads the dataset (e.g., `AI-Driven Sentiment Prediction and Customer Insights Platform.csv`) and preprocesses customer reviews by removing stopwords, lemmatizing words, and tokenizing the text for analysis.

### 2. Sentiment Analysis
Using **VADER** (Valence Aware Dictionary and sEntiment Reasoner), the platform performs sentiment analysis to classify customer reviews into positive, negative, or neutral categories. The sentiment scores are used for further predictive modeling.

### 3. Customer Rating Prediction (SVM)
A **Support Vector Machine (SVM)** model is employed to predict customer ratings based on sentiment scores. The model undergoes hyperparameter tuning using **GridSearchCV** for optimal performance. Evaluation metrics such as **R-squared**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)** are provided.

### 4. Time Series Forecasting
Using **ARIMA** (AutoRegressive Integrated Moving Average), the platform forecasts future customer sentiment trends. This can help businesses proactively address potential customer satisfaction issues and improve customer experience.

### 5. Data Visualization
Interactive visualizations using **Plotly** allow for dynamic exploration of sentiment trends and customer insights:
- **Historical Sentiment Trends**: Line chart visualizing sentiment changes over time.
- **Sentiment Distribution**: Pie chart showing the proportion of positive, neutral, and negative sentiments.

### 6. AI-Driven Business Insights
The platform generates actionable business recommendations based on customer sentiment trends, enabling businesses to:
- Engage satisfied customers with loyalty programs.
- Address negative feedback through root cause analysis and process improvements.
- Reach out to neutral customers for further engagement.

## Dataset
The platform accepts datasets in CSV format, where customer reviews and ratings are the key features. Example columns include:
- **Customer_ID**: Unique identifier for each customer.
- **Review_Text**: The customer's feedback or review.
- **Review_Rating**: The rating provided by the customer (e.g., 1-5).
- **Date**: Date of the review.

## How to Use

### 1. Install Required Dependencies
Install the necessary Python libraries:
```
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Ensure your dataset contains the relevant columns (`Customer_ID`, `Review_Text`, `Review_Rating`, `Date`) and is in CSV format. Rename it to `AI-Driven Sentiment Prediction and Customer Insights Platform.csv` and place it in the root directory.

### 3. Run the Script
Once the dataset is prepared, run the script to perform sentiment analysis, predict customer ratings, and forecast future sentiment trends:
```
python AI-Driven_Sentiment_Prediction_and_Customer_Insights_Platform_Python Script.py
```

### 4. Analyze the Output
The platform generates:
- Predicted customer ratings based on sentiment.
- Actionable insights for improving customer satisfaction.
- Detailed visualizations for exploring sentiment trends.

### 5. Scaling with Big Data (Optional)
If you're dealing with large datasets, the platform can be scaled using **PySpark** for distributed processing. Instructions for scaling are included in the script.

## Libraries Used
- **Pandas**: For data manipulation and preprocessing.
- **NLTK**: For natural language processing and sentiment analysis.
- **Scikit-learn**: For machine learning (SVM) and model evaluation.
- **Statsmodels**: For time series forecasting (ARIMA).
- **Plotly**: For interactive data visualizations.

## Contribution
Contributions are welcome! 

## License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute this platform, provided that proper credit is given.
