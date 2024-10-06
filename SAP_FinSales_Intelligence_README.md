
# SAP FinSales Intelligence Script

## Overview

The **SAP FinSales Intelligence Script** is designed to analyze and optimize **financial performance** (SAP FI/CO) and **sales performance** (SAP SD) data. This script connects directly to **SAP systems** using **pyrfc** for data extraction, and then applies **machine learning techniques** (Random Forest) to forecast profits and evaluate sales outcomes. The results are visualized and accompanied by actionable recommendations to help businesses make data-driven decisions.

### Use Cases:
- **Financial Performance Analysis:** Get insights into revenue, expenses, and profit from SAP FI/CO data.
- **Sales Performance Analysis:** Analyze sales trends and sales amounts from SAP SD data.
- **Profit Forecasting:** Forecast future profit trends using machine learning models.
- **Actionable Recommendations:** Get data-driven suggestions to improve financial outcomes and sales efficiency.

## Key Features

### 1. **SAP Integration**
The script connects to **SAP ERP** systems to extract financial and sales data. Using the **pyrfc** library, the script securely connects to SAP via **RFC (Remote Function Call)** to retrieve data from the **FI/CO** and **SD** modules.

- **SAP FI/CO**: Retrieves financial data like revenue and expenses.
- **SAP SD**: Retrieves sales data such as sales orders, sales amounts, and billing/shipping information.

```python
from pyrfc import Connection
conn = Connection(user='your_username', passwd='your_password', ashost='sap_server', sysnr='00', client='100')
```

### 2. **Data Preprocessing and Error Handling**
Once the data is extracted from SAP, the script handles **missing values** and ensures the data is cleaned for analysis.

- The script uses **pandas** to handle missing values and ensure that numeric columns like **Revenue** and **Expenses** are in the correct format for calculations.
  
```python
df_fi_co['REVENUE'] = pd.to_numeric(df_fi_co['REVENUE'], errors='coerce')
df_fi_co['EXPENSES'] = pd.to_numeric(df_fi_co['EXPENSES'], errors='coerce')
df_fi_co['PROFIT'] = df_fi_co['REVENUE'] - df_fi_co['EXPENSES']
```

- It also merges the financial and sales data based on the common key `SALES_ORDER` to create a unified dataset.

```python
df_merged = pd.merge(df_fi_co, df_sd, left_on='SALES_ORDER', right_on='SALES_ORDER', how='inner')
```

### 3. **Machine Learning for Profit Forecasting**
The core functionality of the script is **profit forecasting** using the **Random Forest Regressor** from **scikit-learn**. This allows businesses to predict future profits based on historical revenue, expenses, and sales performance.

- The script trains a **Random Forest model** using features such as revenue, expenses, and sales amount.

```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

- It then predicts future profits and compares the predicted values with actual profit values for evaluation.

```python
y_pred = rf_model.predict(X_test)
```

### 4. **Visualizations**
The script generates a visualization that compares **actual profit** and **forecasted profit**. This helps businesses evaluate how well the model is predicting financial performance.

- The plot is a **comparison line chart** showing actual vs. forecasted profits over different data points.

```python
plt.plot(y_test.values, label='Actual Profit', color='blue')
plt.plot(y_pred, label='Forecasted Profit', linestyle='--', color='red')
plt.title('Actual vs Forecasted Profit')
```

### 5. **Sales Efficiency and Regional Analysis**
The script also provides insights into **sales efficiency**, such as the time it takes to fulfill sales orders (from billing to shipping). This is calculated by comparing the **shipping date** and **billing date**.

- It also calculates the average fulfillment time and top-performing regions by sales amount.

```python
df_merged['FULFILLMENT_DAYS'] = (pd.to_datetime(df_merged['SHIPPING_DATE']) - pd.to_datetime(df_merged['BILLING_DATE'])).dt.days
```

### 6. **Automatic Interpretation & Recommendations**
The script automatically interprets the results and provides **recommendations** based on key metrics such as:
- Declining profit trends.
- Long fulfillment times.
- Underperforming regions or sales teams.

```python
recommendations = automatic_interpreter(df_merged, y_pred, avg_fulfillment_days, top_regions)
```

### 7. **Automated Reporting**
The script sends an **email report** that includes:
- Predicted profit vs. actual profit.
- Average fulfillment time.
- Top-performing regions.
- Data-driven recommendations to improve sales and profitability.

```python
msg = MIMEText(report_message)
server.sendmail(email_user, ['stakeholder@example.com'], msg.as_string())
```

### 8. **Security and Data Protection**
Sensitive information, such as SAP credentials and email login data, is handled securely through **environment variables**. This ensures that no sensitive information is hard-coded into the script, making it safe for use in production environments.

```python
email_user = os.getenv('EMAIL_USER')
email_password = os.getenv('EMAIL_PASSWORD')
```

## How to Use

### 1. **Install Required Dependencies**
Ensure you have the following Python libraries installed:
```bash
pip install pandas matplotlib scikit-learn pyrfc
```

### 2. **Configure SAP Connection**
Ensure that you have the correct credentials and SAP system parameters to connect to **SAP FI/CO** and **SAP SD** modules.

### 3. **Run the Script**
After setting up the SAP connection and installing the dependencies, run the script to:
- Predict future profits.
- Analyze regional sales and fulfillment efficiency.
- Generate actionable recommendations.

### 4. **Review the Outputs**
The script will generate:
- A **visualization** comparing actual vs. forecasted profits.
- A report showing **top-performing regions** and **sales efficiency**.
- Detailed **recommendations** for improving financial and sales performance.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **matplotlib**: For visualizations.
- **scikit-learn**: For machine learning models.
- **pyrfc**: For SAP data extraction.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.

