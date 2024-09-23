
# AI-Powered Project Management Intelligence Suite (AIPMIS) Script

## Overview

The AI-Powered Project Management Intelligence Suite (AIPMIS) Script is designed to analyze and optimize project management efficiency across multiple key project performance indicators. It leverages machine learning models, including the Random Forest Regressor, to predict project success metrics such as Progress Percentage. Additionally, it provides detailed automatic interpretations and strategies based on key project indicators like Cost Efficiency, Client Satisfaction, and Resource Efficiency, while accounting for project risks.

## Cloud Integration

This script can also be connected to cloud platforms such as AWS, Azure, or Google Cloud for data storage, scalability, and integration with cloud-based AI/ML services. Cloud storage services like AWS S3, Azure Blob, or Google Cloud Storage can be used for handling larger datasets, while services like AWS SageMaker, Azure Machine Learning, or Google AI can be integrated for scalable model training, deployment, and real-time predictions.

## Key Features Include:

- **Project Performance Prediction:** The script predicts project progress using the Random Forest Regressor model based on budget, team size, risks, and other project-specific features.
- **Cost Efficiency & Risk-Adjusted Performance Analysis:** Automatically calculates cost efficiency and risk-adjusted performance for each project, providing strategies for optimizing project management and resource allocation.
- **Detailed Visualizations & Interpretation:** Generates graphical representations of cost efficiency, profit per team member, client satisfaction, and more, with business-friendly interpretations.
- **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to explain the contribution of each project feature to the predicted project outcomes.
- **Professional Summary:** Provides a detailed summary of project performance, including strategies for improving client satisfaction, managing high-risk projects, and optimizing resources.

## Key Features & Functionality

### 1. Data Loading and Error Handling
The script loads project management data from a CSV file and handles missing values by filling them with the mean for numerical columns and the mode for categorical columns. This ensures that the dataset is complete and ready for analysis.

### 2. Project Performance Prediction
The script uses a Random Forest Regressor to predict project progress percentages. It takes into account multiple project metrics, including:

- Budget (USD)
- Actual Cost (USD)
- Team Size
- Risks
- Issues Count
- Client Satisfaction (1-10)
- Resource Utilization (%)
- Duration (Days)

The model predicts how well the project is progressing, and its accuracy is evaluated using Root Mean Squared Error (RMSE).

### 3. Explainable AI with SHAP
The script integrates SHAP to provide explainable AI for the predictions. The SHAP analysis shows how each feature, such as Budget or Team Size, contributes to the overall project progress predictions. This transparency allows project managers to understand which factors are most critical to project success.

### 4. Cost Efficiency & Risk-Adjusted Performance Analysis
The script automatically calculates key project performance metrics, including:

- **Cost Efficiency:** The ratio of the project budget to the actual cost, helping project managers identify under or over-budget projects.
- **Risk-Adjusted Performance:** A measure of project performance that accounts for the risks involved, giving a clearer picture of how well a project is progressing given its risk profile.

### 5. Resource Efficiency and Profitability Analysis
The script provides insights into how efficiently resources are being used on each project and how much profit is being generated per team member. This helps management identify bottlenecks and optimize team size and resource allocation.

### 6. Visualizations
The script generates detailed visualizations for:

- **Cost Efficiency Distribution:** Shows how efficiently projects are using their budgets.
- **Profit per Team Member:** Analyzes profitability based on team size and project performance.
- **Client Satisfaction Distribution:** Identifies projects with high or low client satisfaction.
- **Risk-Adjusted Performance:** Visualizes project progress relative to the risk involved.
- **Resource Efficiency Score:** Shows how effectively resources are being used on each project.

### 7. Automatic Interpretation & Recommendations
For each metric (e.g., cost efficiency, client satisfaction, resource efficiency), the script automatically generates detailed business interpretations and recommendations. This helps project managers understand the current state of their projects and provides strategies for improvement.

### 8. Professional Summary & Next Steps
The script concludes with a high-level executive summary, which includes:

- **Project performance:** A summary of average project efficiency, risk-adjusted performance, and client satisfaction.
- **Strategies & Recommendations:** Based on key performance indicators, the script suggests actionable steps to improve client satisfaction, optimize resource allocation, and address high-risk projects.

## Dataset

The dataset used in this analysis includes the following columns:

- ProjectID: Unique identifier for each project.
- ProjectName: The name of the project.
- StartDate: The project start date.
- EndDate: The project end date.
- Budget (USD): The budget allocated for the project.
- ActualCost (USD): The actual cost incurred by the project.
- TeamSize: The number of team members working on the project.
- ProgressPercentage (%): The current progress of the project, in percentage.
- Risks: A categorical assessment of project risks (e.g., high, medium, low).
- Issues (count): The number of issues or blockers encountered during the project.
- ClientSatisfaction (1-10): A client satisfaction rating for the project (1-10 scale).
- Status: The current status of the project (e.g., completed, in progress, delayed).
- Resource Utilization (%): The percentage of resources being utilized effectively for the project.
- Profitability (USD): The profit generated by the project.

These columns help analyze project performance, including budget adherence, risk management, and client satisfaction.

## How to Use

1. **Install Required Dependencies**

Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

2. **Prepare Your Dataset**
Place your project dataset in the root directory with the name `project_data.csv`.

3. **Run the Script**

Execute the script to:

- **Predict Project Performance:** Use the Random Forest Regressor to predict project progress and analyze key project performance metrics.
- **Analyze Cost Efficiency & Risk:** Automatically calculate cost efficiency, risk-adjusted performance, and resource efficiency.
- **Generate Business Strategies:** Automatically receive recommendations based on project performance.

4. **Review the Outputs**

The script will generate:

- **Cost Efficiency Analysis:** A visualization of how well projects are adhering to their budgets, with automatic interpretations and suggestions.
- **Profit per Team Member:** An analysis of project profitability based on team size.
- **Client Satisfaction Analysis:** A detailed view of client satisfaction across projects and strategies for improvement.
- **SHAP Explanations:** Visual explanations of which factors are most influencing project progress.

## Libraries Used

- **Pandas:** For data manipulation and analysis.
- **Numpy:** For numerical computations.
- **Matplotlib & Seaborn:** For generating visualizations.
- **scikit-learn:** For machine learning model development and evaluation.
- **SHAP:** For explaining machine learning model predictions and providing transparency.

## Contribution

Contributions are welcome!

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.
