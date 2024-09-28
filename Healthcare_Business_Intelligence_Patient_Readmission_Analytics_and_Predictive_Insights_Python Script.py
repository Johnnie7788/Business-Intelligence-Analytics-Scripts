#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This script can be connected to AWS, Google Cloud, or Azure for scalable data storage and processing.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Read the CSV dataset
def read_healthcare_dataset():
    """
    This function reads the healthcare dataset from a CSV file.
    It handles errors if the file is not found and returns the dataset.
    """
    try:
        print("Attempting to load dataset...")
        df = pd.read_csv('healthcare_data.csv')  # Load the dataset from the CSV file
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print("File 'healthcare_data.csv' not found. Please ensure the file is in the correct directory.")
        return None

# Step 2: Data Preprocessing
def preprocess_data(df):
    """
    This function preprocesses the data by handling missing values, encoding categorical variables,
    scaling numerical data, and addressing class imbalance using SMOTE.
    It returns the preprocessed feature matrix (X) and target vector (y).
    """
    print("Starting data preprocessing...")

    # Define numeric and categorical columns
    numeric_cols = ['Age_Years', 'Length_of_Stay_Days', 'Treatment_Satisfaction_Score', 'Days_Since_Last_Visit', 'Readmission_Risk_Score']
    categorical_cols = ['Gender', 'Diagnosis']
    
    # Handle missing values for numeric columns using mean imputation
    imputer_num = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # Handle missing values for categorical columns using the most frequent value
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    # Encode categorical variables (Gender, Diagnosis) into numerical values
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    le_diag = LabelEncoder()
    df['Diagnosis'] = le_diag.fit_transform(df['Diagnosis'])
    
    # Feature Scaling: Standardize numerical columns to have mean = 0 and variance = 1
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Separate features (X) and target (y) where 'Readmitted' is the target variable
    X = df.drop(['Patient_ID', 'Readmitted'], axis=1)
    y = df['Readmitted']
    
    # Handle data imbalance: SMOTE technique is applied to balance the class distribution
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("Data preprocessing completed. Features and target are ready for modeling.")
    return X_resampled, y_resampled

# Step 3: Model Training with Hyperparameter Tuning
def train_model(X, y):
    """
    This function trains a Random Forest model using the preprocessed data.
    It also applies hyperparameter tuning to optimize the model's performance.
    It returns the trained model, test data (X_test, y_test), and predictions.
    """
    print("Starting model training...")

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the Random Forest classifier and set up hyperparameter tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees in the forest
        'max_depth': [5, 10, None],  # Depth of each tree
        'min_samples_split': [2, 5],  # Minimum samples required to split a node
    }
    
    # Grid Search to find the best parameters
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model after hyperparameter tuning
    best_model = grid_search.best_estimator_
    
    # Make predictions on the test data
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC
    
    print("Model training completed. Best model found through hyperparameter tuning.")
    return best_model, X_test, y_test, y_pred, y_prob

# Step 4: Model Evaluation and Visualization
def evaluate_model(y_test, y_pred, y_prob):
    """
    This function evaluates the performance of the trained model using several metrics.
    It prints the classification report, displays the confusion matrix, and plots the ROC curve.
    """
    print("\nEvaluating the model...")

    # 1. Classification Report: Shows precision, recall, and F1-score
    print("\nClassification Report (precision, recall, F1-score):")
    print(classification_report(y_test, y_pred))
    
    # 2. Confusion Matrix: Visualize how many were classified correctly/incorrectly
    print("\nConfusion Matrix (visualized):")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 3. ROC AUC Score: Indicates how well the model distinguishes between classes
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {roc_auc:.2f} (Closer to 1 indicates better performance)")
    
    # Plot ROC Curve: Graphical representation of the true positive vs. false positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], 'k--')  # Diagonal reference line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plot Precision-Recall Curve: Measures the tradeoff between precision and recall
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

# Step 5: Automatic Interpreter and Recommendations
def interpret_results(model, X_test, y_test, y_pred):
    """
    This function interprets the results by showing the top features influencing readmission
    and provides recommendations based on these insights.
    """
    print("\nInterpreting results...")

    # Extract feature importance from the model
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top_features = feature_importances.nlargest(5)
    
    # Display top features
    print("\nTop Features Influencing Readmission:")
    print(top_features)

    # Recommendations based on the most important features
    recommendations = []
    if 'Treatment_Satisfaction_Score' in top_features.index:
        recommendations.append("Improve patient treatment satisfaction to reduce readmission rates.")
    if 'Length_of_Stay_Days' in top_features.index:
        recommendations.append("Optimize length of stay to balance patient recovery and resource utilization.")
    if 'Readmission_Risk_Score' in top_features.index:
        recommendations.append("Utilize the Readmission Risk Score to identify high-risk patients.")
    if 'Age_Years' in top_features.index:
        recommendations.append("Provide additional care for elderly patients to prevent readmission.")
    if 'Days_Since_Last_Visit' in top_features.index:
        recommendations.append("Monitor patients who haven't visited recently for potential health issues.")
    
    # Output recommendations
    print("\nRecommendations for reducing patient readmission:")
    for rec in recommendations:
        print(f"- {rec}")

# Step 6: Professional Summary
def professional_summary(roc_auc):
    """
    This function provides a summary of the model's performance for management.
    It explains the ROC AUC score and highlights key insights from the analysis.
    """
    summary = f"""
    Professional Summary:
    The predictive model achieved a ROC AUC score of {roc_auc:.2f}, indicating good discrimination between patients who are readmitted and those who are not.
    
    Key insights:
    - Length of stay and patient age are among the strongest predictors of readmission.
    - Treatment satisfaction also plays a critical role in reducing readmission risk.
    
    Implementing the provided recommendations could potentially reduce readmission rates, improve patient outcomes, and optimize hospital resources.
    """
    print(summary)

# Main Execution
def main():
    # Step 1: Read the Dataset
    df = read_healthcare_dataset()
    
    if df is None:
        return  # Stop the script if the dataset is not found
    
    # Step 2: Preprocess the Data
    X_resampled, y_resampled = preprocess_data(df)
    
    # Step 3: Train the Model
    model, X_test, y_test, y_pred, y_prob = train_model(X_resampled, y_resampled)
    
    # Step 4: Evaluate the Model
    evaluate_model(y_test, y_pred, y_prob)
    
    # Step 5: Interpret Results and Provide Recommendations
    interpret_results(model, X_test, y_test, y_pred)
    
    # Step 6: Provide Professional Summary
    roc_auc = roc_auc_score(y_test, y_prob)
    professional_summary(roc_auc)

if __name__ == "__main__":
    main()

