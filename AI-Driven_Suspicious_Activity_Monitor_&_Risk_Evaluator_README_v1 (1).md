
# AI-Driven Suspicious Activity Monitor & Risk Evaluator

## Overview

The AI-Driven Suspicious Activity Monitor & Risk Evaluator script is an advanced tool designed to detect and evaluate potentially suspicious financial transactions. By leveraging machine learning techniques such as Random Forest for predictive analysis and Isolation Forest for anomaly detection, this script empowers businesses to identify high-risk activities and take proactive measures in anti-money laundering (AML) efforts. The tool not only highlights suspicious behavior but also provides actionable recommendations, helping organizations enhance their financial security and compliance protocols.

## Key Features Include:

- **Suspicious Activity Prediction:** Uses Random Forest Classifier to predict suspicious transactions based on transaction amount, customer risk score, and transaction frequency.
- **Anomaly Detection:** Implements Isolation Forest to detect unusual transactions that may not have been flagged but exhibit suspicious patterns.
- **Risk Evaluation & Recommendations:** Automatically evaluates customer risk based on their behavior and provides detailed recommendations for further actions.
- **Data Imbalance Handling:** Employs SMOTE to balance the dataset and ensure robust performance across various customer risk profiles.
- **Detailed Visualizations:** Generates visualizations of transactions across countries, risk scores, and suspicious flags to identify trends and high-risk areas.

## Key Features & Functionality

### 1. Data Loading and Error Handling

The script loads the dataset from a CSV file and automatically handles missing values by filling them with the mean for numerical columns. This ensures that the dataset is clean and ready for analysis.

### 2. Suspicious Activity Prediction (Random Forest)

The script uses a Random Forest Classifier to predict whether a transaction is suspicious based on:

- Transaction Amount (EUR)
- Customer Risk Score
- Transaction Frequency

The model provides predictions that help identify suspicious transactions, which are flagged for further review.

### 3. Anomaly Detection (Isolation Forest)

The script applies Isolation Forest to detect outliers or anomalies in the dataset that may indicate fraudulent behavior. This is especially useful for identifying unusual transactions that do not follow regular patterns.

### 4. Risk Evaluation & Recommendations

The script automatically interprets each transaction and provides recommendations based on the following factors:

- **High-Risk Customers:** Flagged for enhanced due diligence and potential reporting to AML teams.
- **Medium-Risk Customers:** Monitored closely for any future suspicious activities.
- **Low-Risk Customers:** No immediate action required.

### 5. Visualizations

The script produces several key visualizations, such as:

- **Transactions by Country:** Shows how transaction amounts differ across countries, highlighting suspicious transactions.
- **Customer Risk Score vs Transaction Amount:** Visualizes the relationship between risk score and transaction amount.
- **Suspicious vs Normal Transactions:** A pie chart showing the proportion of suspicious and normal transactions.

## Dataset

The dataset used in this analysis includes the following columns:

- **Transaction ID:** Unique identifier for each transaction.
- **Transaction Amount (EUR):** The amount of the transaction in euros.
- **Customer Risk Score:** A risk score assigned to each customer based on their historical behavior.
- **Transaction Frequency:** The frequency of transactions by the customer.
- **Country:** The country where the transaction occurred.
- **Transaction Type:** The type of transaction (e.g., online purchase, wire transfer).
- **Suspicious Flag:** A binary flag indicating whether the transaction was previously flagged as suspicious.

These columns help analyze transaction behavior across multiple dimensions, making it easier to identify potential fraud.

## How to Use

### 1. Install Required Dependencies

Ensure you have the following Python libraries installed:
```bash
pip install pandas plotly dash scikit-learn imbalanced-learn
```

### 2. Prepare Your Dataset

Place your dataset in the root directory with the name `suspicious_transactions.csv`.

### 3. Run the Script

Execute the script to:

- **Predict Suspicious Transactions:** Use the Random Forest Classifier to predict suspicious transactions based on customer behavior and transaction attributes.
- **Detect Anomalies:** Use Isolation Forest to flag transactions that exhibit unusual behavior patterns.
- **Generate Business Strategies:** Automatically receive recommendations based on transaction risk and customer behavior.

### 4. Review the Outputs

The script will generate:

- **Transaction Analysis by Country:** Visual representation of suspicious transactions across different countries.
- **Risk Analysis by Customer:** Relationship between customer risk scores and transaction amounts.
- **Anomaly Detection Results:** Flagged anomalous transactions that require further investigation.

## Libraries Used

- **Pandas:** For data manipulation and analysis.
- **Plotly & Dash:** For generating visualizations and building the dashboard.
- **scikit-learn:** For machine learning models like Random Forest and Isolation Forest.
- **imblearn (SMOTE):** For handling data imbalance.

## Contribution

Contributions are welcome! 

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.
