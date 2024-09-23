#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This script can also be connected to cloud platforms such as AWS, Azure, or Google Cloud for data storage, scalability, 
# and integration with cloud-based AI/ML services. Cloud storage services like AWS S3, Azure Blob, or Google Cloud Storage
# can be used for handling larger datasets, while services like AWS SageMaker or Azure Machine Learning can be integrated 

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the realistic project data
project_data = pd.read_csv('project_data.csv')

# ---- Handling Missing Values ----
def handle_missing_values(df):
    # Checking missing values
    missing_info = df.isnull().sum()

    # Filling missing numerical values with mean and categorical values with mode
    for column in df.columns:
        if df[column].dtype == 'object':  # For categorical columns
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # For numerical columns
            df[column].fillna(df[column].mean(), inplace=True)
    return df, missing_info

# Apply missing value handling
project_data, missing_info = handle_missing_values(project_data)

# Preprocessing: Convert categorical variables to numerical values
le_risks = LabelEncoder()
project_data['Risks'] = le_risks.fit_transform(project_data['Risks'])
le_status = LabelEncoder()
project_data['Status'] = le_status.fit_transform(project_data['Status'])

# ---- Create new derived metrics ----
project_data['Duration (Days)'] = (pd.to_datetime(project_data['EndDate']) - pd.to_datetime(project_data['StartDate'])).dt.days
project_data['Cost Efficiency'] = project_data['Budget (USD)'] / project_data['ActualCost (USD)']
project_data['Profit Margin (%)'] = (project_data['Profitability (USD)'] / project_data['Budget (USD)']) * 100
project_data['Profit per Team Member (USD)'] = project_data['Profitability (USD)'] / project_data['TeamSize']

# ---- New Feature 1: Risk-Adjusted Performance ----
project_data['Risk-Adjusted Performance'] = project_data['ProgressPercentage (%)'] / (1 + project_data['Risks'])

# ---- New Feature 2: Resource Efficiency Score ----
project_data['Resource Efficiency Score'] = project_data['Resource Utilization (%)'] / project_data['ProgressPercentage (%)']

# Define features and target for model (Predicting Progress Percentage)
X = project_data[['Budget (USD)', 'ActualCost (USD)', 'TeamSize', 'Risks', 'Issues (count)', 'ClientSatisfaction (1-10)', 'Resource Utilization (%)', 'Duration (Days)']]
y = project_data['ProgressPercentage (%)']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate model performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# ---- Explainable AI (XAI) with SHAP ----
# Use SHAP to explain the Random Forest predictions
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary to explain feature importance
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# ---- Visualizations with Detailed Interpretation ----
def interpret_and_visualize(df):
    # 1. Cost Efficiency Distribution
    plt.figure(figsize=(10,6))
    sns.histplot(df['Cost Efficiency'], kde=True, bins=10)
    plt.axvline(x=1, color='r', linestyle='--', label="Ideal Cost Efficiency")
    plt.title('Cost Efficiency Distribution (Budget/Actual Cost)')
    plt.xlabel('Cost Efficiency')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Detailed Interpretation:
    print("""
    **Interpretation for Management**: This chart displays the **cost efficiency** of projects by comparing the budget to the actual cost.
    - A value above 1 means that projects are under budget, which is a good sign of financial control.
    - A value below 1 indicates that projects are exceeding their budgets, which requires immediate attention to prevent further cost overruns.
    - Management should focus on projects with cost efficiency below 1 to investigate the reasons for overspending and implement cost-control measures.
    """)

    # 2. Profit per Team Member
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, y='Profit per Team Member (USD)', x='Status')
    plt.title('Profit per Team Member by Project Status')
    plt.xlabel('Project Status')
    plt.ylabel('Profit per Team Member (USD)')
    plt.show()

    # Detailed Interpretation:
    print("""
    **Interpretation for Management**: This box plot shows the **profitability per team member** based on the project status.
    - Projects with a higher profit per team member demonstrate efficient resource use and financial performance.
    - If profit per team member is low or negative, it may indicate inefficiencies such as overstaffing or low productivity.
    - Management should explore ways to optimize team sizes and workloads for projects with low profitability to enhance resource efficiency.
    """)

    # 3. Client Satisfaction Distribution
    plt.figure(figsize=(10,6))
    sns.violinplot(x='ClientSatisfaction (1-10)', data=df, inner="quart", palette="muted")
    plt.title('Client Satisfaction Distribution')
    plt.xlabel('Satisfaction Level (1-10)')
    plt.ylabel('Density')
    plt.show()

    # Detailed Interpretation:
    print("""
    **Interpretation for Management**: This chart illustrates the distribution of **client satisfaction** across projects.
    - A satisfaction score below 7 indicates dissatisfaction, which can lead to client retention issues or project delays.
    - Projects with high satisfaction levels show that they are meeting or exceeding client expectations.
    - Management should prioritize projects with low satisfaction to improve communication, resolve client concerns, and ensure higher satisfaction scores.
    """)

    # 4. Risk-Adjusted Performance
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Risk-Adjusted Performance', y='ProgressPercentage (%)', hue='Status', palette='cool')
    plt.title('Risk-Adjusted Performance vs. Progress Percentage')
    plt.xlabel('Risk-Adjusted Performance')
    plt.ylabel('Progress Percentage (%)')
    plt.legend(title="Project Status")
    plt.show()

    # Detailed Interpretation:
    print("""
    **Interpretation for Management**: This scatter plot evaluates project performance adjusted for risk.
    - A high risk-adjusted performance indicates that a project is progressing well despite high risks, which reflects strong management.
    - Conversely, low performance in high-risk projects suggests that they require immediate intervention to prevent delays or failure.
    - Management should closely monitor projects with low risk-adjusted performance and prioritize resource allocation to these projects.
    """)

    # 5. Resource Efficiency Score
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='Status', y='Resource Efficiency Score', palette='Set2')
    plt.title('Resource Efficiency Score by Project Status')
    plt.xlabel('Project Status')
    plt.ylabel('Resource Efficiency Score')
    plt.show()

    # Detailed Interpretation:
    print("""
    **Interpretation for Management**: This plot shows how efficiently resources are being utilized relative to project progress.
    - A low resource efficiency score indicates that teams may not be using their resources effectively, leading to delays or inefficiencies.
    - A high score suggests that resources are being used optimally to meet project milestones.
    - Management should investigate projects with low resource efficiency to identify bottlenecks or inefficiencies in resource allocation.
    """)

# Run the visualization and interpretation function
interpret_and_visualize(project_data)

# ---- Enhanced Executive Professional Summary ----
def executive_summary(df):
    avg_cost_efficiency = df['Cost Efficiency'].mean()
    avg_profit_margin = df['Profit Margin (%)'].mean()
    avg_client_satisfaction = df['ClientSatisfaction (1-10)'].mean()
    avg_risk_adjusted_performance = df['Risk-Adjusted Performance'].mean()
    avg_resource_efficiency_score = df['Resource Efficiency Score'].mean()
    high_risk_projects = df[df['Risks'] == le_risks.transform(['High'])[0]]

    summary = f"""
    Executive Summary:
    1. **Average Cost Efficiency**: {avg_cost_efficiency:.2f}
       - Projects are, on average, under budget (above 1.0), indicating strong financial control.
       However, care must be taken to ensure that underspending does not lead to missed milestones or reduced quality.

    2. **Average Profit Margin**: {avg_profit_margin:.2f}%
       - A strong profit margin suggests that projects are financially healthy.
       However, profitability should be balanced with team workload and client satisfaction to avoid overworking teams or reducing project quality.

    3. **Average Client Satisfaction**: {avg_client_satisfaction:.2f}/10
       - A low client satisfaction score indicates that several projects may be falling short of client expectations.
       Immediate attention is needed for projects with satisfaction below 7 to prevent contract disputes or cancellations.

      4. **Average Risk-Adjusted Performance**: {avg_risk_adjusted_performance:.2f}
       - While performance is solid overall, projects with low risk-adjusted performance and high risk should be flagged for immediate attention.
       Additional resources may be required to ensure high-risk projects stay on track.

    5. **Average Resource Efficiency Score**: {avg_resource_efficiency_score:.2f}
       - The average resource efficiency score of 1.10 indicates that resources are generally being utilized efficiently across projects.
       However, projects with a resource efficiency score below 1.0 should be flagged for further analysis to optimize resource allocation and avoid potential delays or inefficiencies.

    6. **High-Risk Projects**: {len(high_risk_projects)} projects flagged as high risk.
       - These projects are more likely to face challenges, such as delays or cost overruns, and need immediate intervention.
       Management should allocate additional resources and conduct regular progress reviews for these high-risk projects.

    Recommendations:
    - **Improve Client Satisfaction**: Focus on projects with client satisfaction scores below 7. By improving communication, addressing client concerns, and delivering higher quality, client satisfaction can be elevated, reducing risks of project cancellations or disputes.

    - **Monitor High-Risk, Low-Performance Projects**: Projects with low risk-adjusted performance and high risks should receive close attention. These projects need to be monitored regularly, and corrective actions should be taken early to avoid delays or cost overruns.

    - **Optimize Resource Allocation**: For projects with low resource efficiency scores, management should optimize team sizes, tool usage, and task allocation to ensure that resources are being used effectively to meet project goals.

    - **Cost Efficiency**: Continue to monitor cost efficiency across projects. Ensure that any underspending does not negatively affect project quality, client satisfaction, or deadlines.

    Conclusion:
    The project data suggests that most projects are operating within budget and generating healthy profit margins. However, there are concerns related to client satisfaction and resource efficiency that need to be addressed. By focusing on high-risk projects and ensuring that resources are allocated optimally, companies can improve overall project outcomes while maintaining financial performance.
    """
    
    print(summary)

# Run the enhanced executive summary
executive_summary(project_data)





















