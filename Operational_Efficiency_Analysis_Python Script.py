#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers

# Load the dataset with units in column names
df = pd.read_csv('Operational_Efficiency_Data.csv')

# Step 1: Handling Missing Data (exclude non-numeric columns like 'Process_ID' and 'Date')
def handle_missing_values(data):
    """Handle missing values using mean imputation for numeric columns."""
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    
    # Combine with non-numeric columns
    non_numeric_data = data.select_dtypes(exclude=[np.number])  # Select non-numeric columns
    data_clean = pd.concat([non_numeric_data.reset_index(drop=True), data_imputed], axis=1)
    
    return data_clean

df_clean = handle_missing_values(df)

# Step 2: Feature Engineering - Scaling the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_clean.drop(['Process_ID', 'Date'], axis=1)), 
                         columns=df_clean.drop(['Process_ID', 'Date'], axis=1).columns)

# Step 3: Check Class Distribution and Address Class Imbalance using SMOTE
X = scaled_df.drop('Process_Efficiency (%)', axis=1)
y = scaled_df['Process_Efficiency (%)']

# Categorize efficiency into binary classes: Efficient (>= 80%) or Inefficient (< 80%)
y_binary = np.where(y >= 80, 1, 0)  # 1 for efficient, 0 for inefficient

# Check if there is more than one class in the target variable
unique_classes, class_counts = np.unique(y_binary, return_counts=True)
print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")

if len(unique_classes) > 1:
    # Apply SMOTE to balance the classes only if there is an imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_binary)
    print(f"After SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
else:
    # If only one class is present, skip SMOTE and use the original data
    print("Skipping SMOTE as only one class is present.")
    X_resampled, y_resampled = X, y_binary

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: AI-driven Prediction Model (Neural Network)
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (efficient or inefficient)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Predict and calculate accuracy
predictions = model.predict(X_test)
predictions_binary = np.where(predictions > 0.5, 1, 0)
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Visualize the Results (Fixing the TypeError in sns.lineplot)
def visualize_efficiency(data):
    """Visualize operational efficiency trends."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data.index, y=data['Process_Efficiency (%)'], marker='o', label='Process Efficiency')
    plt.title('Operational Efficiency Over Time')
    plt.xlabel('Instance')
    plt.ylabel('Efficiency (%)')
    plt.legend()
    plt.show()

visualize_efficiency(df_clean)

# Step 7: Automatic Interpretation & Recommendations
def automatic_interpreter(data, predictions):
    """Automatically interpret visualizations and provide recommendations."""
    avg_efficiency = np.mean(data['Process_Efficiency (%)'])
    latest_efficiency = predictions[-1][0] * 100  # Convert sigmoid output to percentage
    improvement_needed = avg_efficiency - latest_efficiency
    
    print("\n--- Automatic Interpretation ---")
    print(f"Average Efficiency: {avg_efficiency:.2f}%")
    print(f"Latest Predicted Efficiency: {latest_efficiency:.2f}%")
    
    if latest_efficiency < avg_efficiency:
        print("Recommendation: Process optimization is required. Focus on reducing task errors, time taken, and resource usage.")
        print("Strategy: Automate repetitive tasks, implement AI-driven scheduling, and allocate resources dynamically based on demand.")
    else:
        print("Recommendation: The process is performing efficiently. Continue monitoring for potential future inefficiencies.")

# Apply the interpreter on predictions
automatic_interpreter(df_clean, predictions)

# Step 8: Addressing Missing Values and Generating Insights
def handle_and_inspect_missing(data):
    """Handle missing values and inspect data quality."""
    missing_values = data.isnull().sum()
    print(f"Missing values in the dataset:\n{missing_values}")
    
    # Handling missing values again
    data_filled = handle_missing_values(data)
    print("Missing values handled using mean imputation.")
    return data_filled

df_final = handle_and_inspect_missing(df)

# Step 9: Generate Professional Summary of Operational Efficiency
def generate_summary(data, predictions, accuracy):
    """Generate a professional summary of operational efficiency."""
    avg_efficiency = np.mean(data['Process_Efficiency (%)'])
    latest_efficiency = predictions[-1][0] * 100  # Convert sigmoid output to percentage
    performance_trend = "improving" if latest_efficiency >= avg_efficiency else "declining"
    
    print("\n--- Professional Summary of Operational Efficiency ---")
    print(f"Overall Process Efficiency: {avg_efficiency:.2f}%")
    print(f"Latest Process Efficiency Prediction: {latest_efficiency:.2f}%")
    print(f"Performance Trend: {performance_trend}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    if performance_trend == "declining":
        print("Areas of Improvement: Focus on reducing process delays, minimizing resource consumption, and cutting down on errors.")
        print("Strategic Recommendation: Implement AI-driven automation and lean operational practices.")
    else:
        print("Current Performance: The process is operating efficiently, continue monitoring and fine-tuning processes as necessary.")
        
# Generate the professional summary
generate_summary(df_clean, predictions, accuracy)










































































