
# Consumer Insights Analysis Suite

## Overview
The **Consumer Insights Analysis Suite** is an advanced Python script developed to help businesses gain actionable insights into consumer behavior and market dynamics. This suite combines machine learning models, sentiment analysis, statistical methods, and advanced visualizations to optimize business strategies and support data-driven decision-making.

This script is designed to assist in key business functions, such as understanding market share dynamics, evaluating ad spend efficiency, predicting future trends, and generating recommendations for marketing adjustments. Ideal for roles focused on consumer and market insights, it offers a 360-degree view of customer needs and behaviors.

### Key Features:
1. **Sentiment-Driven Sales Prediction**: Predicts future sales based on customer sentiment, purchasing behavior, and market performance metrics.
2. **Dynamic Market Share Adjustment**: Provides real-time adjustments to market share based on competition and performance data.
3. **Ad Spend Efficiency Index (ASEI)**: Calculates the efficiency of advertising spend, measuring its impact on sales.
4. **Time Series Clustering**: Clusters products or customer segments based on time-series data, including sales, sentiment, and ad spend, to identify hidden patterns.
5. **Next-Best Action Recommendations**: Automatically generates recommendations to adjust ad spend or marketing strategies for each customer segment based on ASEI and performance insights.
6. **AI-Driven Business Insights**: Interprets visualizations and key metrics, offering strategic insights for stakeholders.
7. **Inferential Statistics**: Quantifies the impact of customer sentiment on purchase behavior through statistical analysis.
8. **Quasi-Experimental Analysis**: Evaluates the effectiveness of marketing campaigns by comparing pre- and post-campaign performance.
9. **Data Visualization**: Produces insightful visualizations to help businesses easily interpret trends, including sales predictions, market share dynamics, and sentiment distribution.

---

## Key Features & Functionality

### 1. Data Loading and Preprocessing
The script loads both survey (perception) data and transactional (behavioral) data from a CSV file and SQL database, respectively. It merges these datasets on a common identifier (e.g., `customer_id`) and manages missing values to ensure data integrity.

### 2. Sentiment-Driven Sales Prediction
Using sentiment scores derived from survey responses, the script predicts future sales volumes with a **Random Forest Regressor** model. The model includes hyperparameter tuning for optimal performance, integrating sentiment scores, ad spend, and other performance indicators.

### 3. Dynamic Market Share Adjustment & ASEI
The script calculates the **Ad Spend Efficiency Index (ASEI)** to measure how effectively ad spending translates into sales. It also dynamically adjusts market share metrics based on real-time data on competition and performance, providing businesses with a clear understanding of their market position.

### 4. Time Series Clustering
By applying **KMeans Clustering** on time-series data, the script identifies patterns within sales, sentiment, and ad spend data. This clustering helps reveal customer segments or product groups that demonstrate similar behavior, which can drive targeted marketing efforts.

### 5. Social Media Sentiment Tracking
The suite simulates real-time sentiment analysis by tracking and scoring customer sentiment. This feature identifies sentiment spikes that may impact sales, offering timely insights for potential PR interventions or marketing adjustments.

### 6. Next-Best Action Recommendations
Based on ASEI, sales performance, and customer sentiment, the suite generates tailored marketing recommendations. It suggests specific actions, such as increasing or decreasing ad spend for each segment, enabling businesses to optimize their strategies in real-time.

### 7. Inferential Statistics
The script conducts **OLS regression** to understand the influence of sentiment on spending. This analysis highlights how customer sentiment correlates with purchase behavior, providing insights that support strategic decision-making.

### 8. Quasi-Experimental Analysis
Using a quasi-experimental design, the script evaluates the effectiveness of marketing campaigns by comparing pre- and post-campaign sales data. This approach helps quantify the impact of promotional efforts and supports evidence-based marketing decisions.

### 9. Data Visualization
Detailed visualizations are produced to facilitate easy interpretation of market trends:
- **Sales Volume vs. Predicted Sales**: Highlights the relationship between actual and forecasted sales volumes.
- **Dynamic Market Share Over Time**: Tracks market share dynamics across customer segments or product categories.
- **Cluster Heatmap**: Displays hidden patterns in sales, sentiment, and ad spend.
- **Sales Volume vs. ASEI**: Shows the relationship between sales and advertising efficiency, identifying high-impact strategies.

---

## Dataset

The suite operates on two main datasets:

1. **Survey Data (e.g., `survey_data.csv`)**:
   - **customer_id**: Unique identifier for each customer.
   - **open_ended_response**: Customer feedback used for sentiment analysis.

2. **Transactional Data (SQL Database)**:
   - **customer_id**: Unique identifier for each customer.
   - **purchase_frequency**: Number of purchases made by each customer.
   - **average_order_value**: Average value of purchases per customer.
   - **total_spent**: Total spending by each customer.
   - **ad_spend**: Total advertising expenditure per customer.
   - **market_share**: Current market share for each product or customer segment.
   - **date**: Transaction date for time-series clustering.

These datasets can be adapted to fit various industries and business needs.

---

## How to Use

### 1. Install Required Dependencies
Install the necessary Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn vaderSentiment sqlalchemy
```

### 2. Prepare Your Datasets
Ensure that survey data is available as a CSV file (e.g., `survey_data.csv`), and transactional data is accessible in a SQL database.

### 3. Run the Script
Execute the script from the command line, specifying file paths and connection strings as arguments:

```bash
python ConsumerInsights_AnalysisSuite.py --survey_filepath path/to/survey_data.csv --sql_connection_string 'sqlite:///path/to/database.db'
```

### 4. Analyze the Output
The script generates:
- **Predicted Sales Volumes**: Based on sentiment scores and ad spend.
- **Market Share Analysis**: Dynamic adjustments of market share in response to performance metrics.
- **Ad Spend Efficiency Index (ASEI)**: Assesses the return on advertising investments.
- **Clustered Insights**: Time series-based clustering of segments or products.
- **Campaign Impact Analysis**: Evaluates the effectiveness of marketing campaigns.
- **Detailed Visualizations**: Saved as PNG files for easy reference.

### 5. Customize the Script
This suite is flexible and can be adapted to fit various datasets, industries, and business needs. Modify model parameters and features as needed to better suit your use case.

---

## Libraries Used
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Clustering (KMeans) and predictive modeling (Random Forest).
- **Statsmodels**: Statistical analysis for regression and impact evaluation.
- **VADER**: Sentiment analysis library for scoring customer feedback.
- **SQLAlchemy**: For SQL database connectivity.
- **Matplotlib and Seaborn**: Visualization libraries.

---

## Contribution
Contributions are welcome! 

---

## License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute this script, provided that proper credit is given.

