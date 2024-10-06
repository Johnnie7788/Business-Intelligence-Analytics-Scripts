
from pyrfc import Connection
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Step 1: SAP Connection and Data Extraction
try:
    conn = Connection(user='your_username', passwd='your_password', ashost='sap_server', sysnr='00', client='100')
    print("Connection successful.")
except Exception as e:
    print(f"Error connecting to SAP: {e}")
    raise

# Query financial data (FI/CO) using RFC
try:
    fi_co_data = conn.call('BAPI_NAME_FOR_FI_CO', {'PARAMETERS': 'your_parameters'})
    df_fi_co = pd.DataFrame(fi_co_data['results'])
except Exception as e:
    print(f"Error fetching FI/CO data: {e}")
    raise

# Query sales data (SD) using RFC
try:
    sd_data = conn.call('BAPI_NAME_FOR_SD', {'PARAMETERS': 'your_parameters'})
    df_sd = pd.DataFrame(sd_data['results'])
except Exception as e:
    print(f"Error fetching SD data: {e}")
    raise

# Close connection after data extraction
conn.close()

# Step 2: Data Preprocessing and Merging
df_fi_co['REVENUE'] = pd.to_numeric(df_fi_co['REVENUE'], errors='coerce')
df_fi_co['EXPENSES'] = pd.to_numeric(df_fi_co['EXPENSES'], errors='coerce')
df_fi_co['PROFIT'] = df_fi_co['REVENUE'] - df_fi_co['EXPENSES']

try:
    df_merged = pd.merge(df_fi_co, df_sd, left_on='SALES_ORDER', right_on='SALES_ORDER', how='inner')
    print("Data successfully merged.")
except KeyError as e:
    print(f"Error in merging data: {e}")
    raise

# Step 3: Advanced Analytics (Sales and Profit Forecasting)
X = df_merged[['REVENUE', 'EXPENSES', 'SALES_AMOUNT']].dropna()
y = df_merged['PROFIT'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(f"Predicted Profit: {y_pred[:5]}")

# Step 4: Visualization
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Profit', color='blue')
plt.plot(y_pred, label='Forecasted Profit', linestyle='--', color='red')
plt.title('Actual vs Forecasted Profit')
plt.xlabel('Data Points')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Step 5: Sales Efficiency and Revenue Impact Analysis
df_merged['SHIPPING_DATE'] = pd.to_datetime(df_merged['SHIPPING_DATE'], errors='coerce')
df_merged['BILLING_DATE'] = pd.to_datetime(df_merged['BILLING_DATE'], errors='coerce')
df_merged['FULFILLMENT_DAYS'] = (df_merged['SHIPPING_DATE'] - df_merged['BILLING_DATE']).dt.days

avg_fulfillment_days = df_merged['FULFILLMENT_DAYS'].mean()
print(f"Average Fulfillment Time: {avg_fulfillment_days} days")

top_regions = df_merged.groupby('SALES_REGION')['SALES_AMOUNT'].sum().sort_values(ascending=False).head(5)
print("Top 5 Sales Regions by Revenue:
", top_regions)

# Step 6: Automated Reporting
def send_email_report_with_recommendations():
    report_message = f"Financial & Sales Performance Report\n\nPredicted Profit: {y_pred[:5]}\nAverage Fulfillment Time: {avg_fulfillment_days} days\nTop Sales Regions: {top_regions}\n\n--- Recommendations ---\n"
    for i, rec in enumerate(recommendations, 1):
        report_message += f"{i}. {rec}\n"
    
    email_user = os.getenv('EMAIL_USER')
    email_password = os.getenv('EMAIL_PASSWORD')
    
    msg = MIMEText(report_message)
    msg['Subject'] = 'SAP FinSales Intelligence Report with Recommendations'
    msg['From'] = email_user
    msg['To'] = 'stakeholder@example.com'

    try:
        with smtplib.SMTP('smtp.example.com') as server:
            server.login(email_user, email_password)
            server.sendmail(email_user, ['stakeholder@example.com'], msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Step 7.1: Automatic Interpreter Logic
def automatic_interpreter(df_merged, y_pred, avg_fulfillment_days, top_regions):
    recommendations = []
    if y_pred.mean() is not None and df_merged['PROFIT'].mean() is not None and y_pred.mean() < df_merged['PROFIT'].mean():
        recommendations.append("Sales forecast shows a declining trend in profits. Consider revising the sales strategy, adjusting pricing, or improving customer engagement to boost sales.")
    
    if avg_fulfillment_days is not None and avg_fulfillment_days > 5:
        recommendations.append(f"Average fulfillment time is {avg_fulfillment_days:.2f} days, which exceeds the 5-day threshold. Consider optimizing the supply chain or streamlining order processing to reduce delays.")
    
    low_performing_regions = df_merged.groupby('SALES_REGION')['SALES_AMOUNT'].sum().sort_values().head(3)
    if not low_performing_regions.empty:
        recommendations.append(f"Sales in the following regions are underperforming: {', '.join(low_performing_regions.index)}. Consider conducting a regional market analysis and adjusting sales efforts to improve performance.")
    
    profit_margin = df_merged['PROFIT'].mean() / df_merged['REVENUE'].mean() * 100 if df_merged['REVENUE'].mean() else None
    if profit_margin is not None and profit_margin < 10:
        recommendations.append(f"The average profit margin is {profit_margin:.2f}%, which is below the optimal range. Consider reducing expenses or optimizing product pricing to improve profitability.")
    
    if not recommendations:
        recommendations.append("All key performance indicators are within optimal ranges. Continue monitoring for potential market changes.")
    
    return recommendations

# Step 7.2: Generate and Display Recommendations
recommendations = automatic_interpreter(df_merged, y_pred, avg_fulfillment_days, top_regions)
print("\n--- Automatic Recommendations ---")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

# Step 7.3: Email with Recommendations
send_email_report_with_recommendations()
