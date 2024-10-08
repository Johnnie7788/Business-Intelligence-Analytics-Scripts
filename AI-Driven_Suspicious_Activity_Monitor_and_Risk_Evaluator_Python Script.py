#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset 
df = pd.read_csv('suspicious_transactions.csv')

# Automatic interpreter for suspicious flags
def interpret_suspicious_flags(row):
    if row['Suspicious_Flag'] == 1:
        return f"Suspicious transaction detected. Customer Risk Score: {row['Customer_Risk_Score']}. Investigate immediately!"
    else:
        return f"Transaction appears normal. Customer Risk Score: {row['Customer_Risk_Score']}."

df['Interpretation'] = df.apply(interpret_suspicious_flags, axis=1)

# Recommendations based on suspicious transactions and risk score
def generate_recommendations(row):
    if row['Suspicious_Flag'] == 1 and row['Customer_Risk_Score'] > 50:
        return "High-risk customer with suspicious activity. Consider enhanced due diligence (EDD) and report to AML team."
    elif row['Suspicious_Flag'] == 1:
        return "Suspicious transaction flagged. Review the transaction details and conduct appropriate checks."
    elif row['Customer_Risk_Score'] > 50:
        return "High-risk customer. Monitor for any future suspicious activity."
    else:
        return "No immediate action required."

df['Recommendations'] = df.apply(generate_recommendations, axis=1)

# Handle missing values
df['Transaction_Amount_EUR'].fillna(df['Transaction_Amount_EUR'].mean(), inplace=True)
df['Transaction_Frequency'].fillna(df['Transaction_Frequency'].mode()[0], inplace=True)

# Handling Data Imbalance with SMOTE
X = df[['Transaction_Amount_EUR', 'Customer_Risk_Score', 'Transaction_Frequency']]
y = df['Suspicious_Flag']

# SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Replace original dataset with resampled data
df_resampled = pd.DataFrame(X_resampled, columns=['Transaction_Amount_EUR', 'Customer_Risk_Score', 'Transaction_Frequency'])
df_resampled['Suspicious_Flag'] = y_resampled

# Add columns back to the resampled dataframe
df_resampled['Country'] = df['Country'][:len(df_resampled)]
df_resampled['Transaction_Type'] = df['Transaction_Type'][:len(df_resampled)]

# Split the dataset for Random Forest training
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict Suspicious Transactions using the test data
y_pred = rf_model.predict(X_test)

# Output Classification Report
print(classification_report(y_test, y_pred))

# Add Random Forest predictions to the dataset
df_resampled['RF_Prediction'] = rf_model.predict(X_resampled)

# Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df_resampled['Anomaly'] = iso_forest.fit_predict(X_resampled)

# Plotly visualization: Transactions by Country
fig_transactions_by_country = px.bar(
    df_resampled, x='Country', y='Transaction_Amount_EUR', color='Suspicious_Flag',
    title="Transactions by Country with Suspicious Flags",
    labels={'Transaction_Amount_EUR': 'Transaction Amount (EUR)', 'Country': 'Country'}
)

# Plotly visualization: Customer Risk Scores vs Suspicious Flags
fig_risk_vs_suspicious = px.scatter(
    df_resampled, x='Customer_Risk_Score', y='Transaction_Amount_EUR', color='Suspicious_Flag',
    title="Customer Risk Score vs Transaction Amount (EUR)",
    labels={'Customer_Risk_Score': 'Customer Risk Score', 'Transaction_Amount_EUR': 'Transaction Amount (EUR)'}
)

# Pie Chart for Suspicious vs Normal Transactions
fig_pie_chart = px.pie(
    df_resampled, names='Suspicious_Flag', title="Distribution of Suspicious vs Normal Transactions",
    labels={'Suspicious_Flag': 'Suspicious Transaction'}
)

# Create Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1("Business Intelligence & Anti-Money Laundering Dashboard"),


    # Interpretation & Recommendations Section
    html.Div(children=[
        html.H3("Interpretation and Recommendations"),
        html.Table([
            html.Tr([html.Th("Transaction ID"), html.Th("Interpretation"), html.Th("Recommendations")]),
            *[html.Tr([html.Td(row['Transaction_ID']), html.Td(row['Interpretation']), html.Td(row['Recommendations'])]) for _, row in df.iterrows()]
        ])
    ]),

    # Plot: Transactions by Country
    html.Div(children=[
        html.H3("Transactions by Country"),
        dcc.Graph(figure=fig_transactions_by_country)
    ]),

    # Plot: Customer Risk Score vs Suspicious Flag
    html.Div(children=[
        html.H3("Customer Risk Score vs Transaction Amount"),
        dcc.Graph(figure=fig_risk_vs_suspicious)
    ]),

    # Pie Chart: Suspicious vs Normal Transactions
    html.Div(children=[
        html.H3("Suspicious vs Normal Transactions"),
        dcc.Graph(figure=fig_pie_chart)
    ]),
])

# Automatically open the app in a browser
def open_browser():
    import webbrowser
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    open_browser()
    app.run_server(debug=True)

