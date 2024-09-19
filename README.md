
# Customer Behavior Analytics & Predictive Insights Script

## Overview

This repository contains a **Business Intelligence and Predictive Analytics Script** designed for comprehensive **Customer Behavior Analysis**. The script employs advanced machine learning models and data processing techniques to provide actionable insights into customer segments, behavior patterns, and personalized recommendations. It is specifically designed to enhance decision-making processes in customer retention, dynamic pricing, and proactive engagement strategies.

Key features include:

1. **Sentiment Analysis** on customer feedback to gauge overall customer sentiment.
2. **Behavioral Segmentation** using clustering techniques for more precise customer targeting.
3. **Dynamic Pricing Models** based on purchase frequency and market conditions.
4. **Churn Prediction** to identify at-risk customers, enabling preemptive retention efforts.
5. **Next-Best Action Recommendations** to suggest personalized actions for improving customer engagement.
6. **AI-Driven Business Insights** for high-level strategic decision-making.

The script is suitable for businesses aiming to optimize customer relationships, improve retention rates, and drive profitability through data-driven strategies.

---

## Key Features & Functionality

### 1. Data Loading and Preprocessing

The script begins by loading the dataset (`customer_data.csv`) and performing missing value imputation using appropriate strategies for both numerical and categorical data. This ensures a clean dataset for further analysis and modeling.

### 2. Sentiment Analysis

Sentiment analysis is performed on customer feedback using the **TextBlob** library, generating a `Sentiment_Score` that measures the polarity of customer feedback. This score ranges from -1 (negative sentiment) to 1 (positive sentiment), providing valuable insights into overall customer satisfaction.

### 3. Behavioral Segmentation

Utilizing **KMeans Clustering**, the script segments customers based on their purchase frequency and engagement score. These segments help in understanding diverse customer behaviors and targeting them more effectively with tailored strategies.

### 4. Dynamic Pricing Model

The script implements a **dynamic pricing algorithm** that adjusts pricing based on customer behavior and current market conditions. This feature is crucial for maximizing revenue by aligning prices with demand and customer willingness to pay.

### 5. Churn Prediction

The **Logistic Regression** model is employed to predict the likelihood of customer churn. The model is fine-tuned through **GridSearchCV** for optimal performance. The script also utilizes **SMOTE** for balancing the dataset, ensuring better model accuracy and reduced bias.

Key Performance Metrics reported include:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### 6. Next-Best Action Recommendations

A **Random Forest Classifier** is used to determine the next-best action for each customer, such as recommending a product or sending a re-engagement offer. This recommendation system is highly beneficial for designing personalized marketing campaigns to improve customer retention and engagement.

### 7. AI-Driven Business Insights

The script generates strategic **AI-driven insights** by analyzing customer segments, churn risk, sentiment distribution, and other key business metrics. These insights are crucial for understanding business dynamics and making data-driven decisions to improve customer relationships and optimize operations.

### 8. Data Visualization

The script produces several detailed visualizations, including:

- **Sentiment Score Distribution**
- **Purchase Frequency vs. Engagement Score by Platform**
- **Purchase Amount Distribution by Platform**
- **Max Price by Preferred Category**

These visualizations provide a clear graphical representation of customer behavior, aiding in the interpretation of complex data trends.

---

## Dataset

This repository includes the first six rows of the customer dataset (`customer_data.csv`). This demonstrates the structure of the data used in the script, which includes:

- **Customer_ID**: A unique identifier for each customer.
- **Purchase_Frequency**: Number of purchases made by the customer within a defined time period.
- **Engagement_Score**: A score representing the level of customer engagement with the business.
- **Feedback_Text**: Textual feedback from customers, analyzed for sentiment.
- **Preferred_Category**: The product category most favored by the customer.
- **Platform**: The platform used for purchases (e.g., mobile app, desktop).
- **Purchase_Amount**: The total monetary amount spent by the customer.
- **Max_Price**: Maximum price the customer is willing to pay for a product or service.

The script is designed to handle a variety of customer data formats and can be adapted to different business needs.

---

## How to Use

### 1. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Rename your dataset to `customer_data.csv` and place it in the root directory. The script is configured to handle CSV files but can be easily modified to accept other formats.

### 3. Run the Script

Once the dataset is prepared, you can run the script:

```bash
python customer_behavior_analysis.py
```

### 4. Analyze the Output

The script generates:

- **Predicted churn risks** and **next-best actions** for each customer.
- **Actionable business insights** for better decision-making.
- **Visualizations** for easier interpretation of customer behavior and trends.

### 5. Adapt the Script

The script is flexible and can be adapted to different datasets and business needs. Users are encouraged to modify the models, parameters, and features according to their specific use cases.

---

## Libraries Used

- **Pandas**: For data manipulation and preprocessing.
- **TextBlob**: For sentiment analysis of customer feedback.
- **scikit-learn**: For clustering (KMeans), predictive modeling (Logistic Regression, Random Forest), and model evaluation metrics.
- **imblearn**: For handling imbalanced datasets using SMOTE.
- **Matplotlib** and **Seaborn**: For data visualization.

---

## Contribution

Contributions are welcome. 

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.
