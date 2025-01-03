#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataQualityGovernance:
    def __init__(self, data_path):
        """Initialize the class with the data path."""
        self.data = pd.read_csv(data_path)
        self.report = {}

    def validate_data_integrity(self):
        """Validate data integrity and identify anomalies."""
        logging.info("Validating data integrity...")
        try:
            # Check for duplicates
            duplicates = self.data.duplicated().sum()
            self.report['duplicates'] = duplicates
            logging.info(f"Number of duplicate rows: {duplicates}")

            # Check for missing values
            missing_values = self.data.isnull().sum()
            missing_percentage = (missing_values / len(self.data)) * 100
            self.report['missing_values'] = missing_values.to_dict()
            logging.info(f"Missing value report:\n{missing_values}")

            # Detect outliers using Isolation Forest
            numeric_data = self.data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                isolation_forest = IsolationForest(random_state=42, contamination=0.05)
                outlier_predictions = isolation_forest.fit_predict(numeric_data)
                outliers = (outlier_predictions == -1).sum()
                self.report['outliers'] = outliers
                logging.info(f"Number of outliers detected: {outliers}")
            else:
                self.report['outliers'] = 0
                logging.info("No numeric data available for outlier detection.")

        except Exception as e:
            logging.error(f"Error during data integrity validation: {e}")
            raise

    def monitor_quality_metrics(self):
        """Monitor and report key data quality metrics."""
        logging.info("Monitoring data quality metrics...")
        try:
            # Completeness
            completeness = 100 - (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100)
            self.report['completeness'] = completeness
            logging.info(f"Data completeness: {completeness:.2f}%")

            # Accuracy (placeholder: requires domain-specific rules)
            # Example: Checking for invalid dates
            if 'date' in self.data.columns:
                invalid_dates = self.data['date'].apply(lambda x: pd.to_datetime(x, errors='coerce')).isnull().sum()
                self.report['invalid_dates'] = invalid_dates
                logging.info(f"Number of invalid dates: {invalid_dates}")

        except Exception as e:
            logging.error(f"Error during quality metrics monitoring: {e}")
            raise

    def error_correction(self):
        """Suggest or implement fixes for data inconsistencies."""
        logging.info("Performing error correction...")
        try:
            # Impute missing values
            imputer = SimpleImputer(strategy='mean')
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = imputer.fit_transform(self.data[numeric_columns])
            logging.info("Missing values imputed with mean values.")

            # Drop duplicate rows
            duplicates_before = len(self.data)
            self.data = self.data.drop_duplicates()
            duplicates_after = len(self.data)
            self.report['duplicates_removed'] = duplicates_before - duplicates_after
            logging.info(f"Duplicates removed: {duplicates_before - duplicates_after}")

            # Handle outliers by capping
            for column in numeric_columns:
                q1 = self.data[column].quantile(0.25)
                q3 = self.data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data[column] = np.clip(self.data[column], lower_bound, upper_bound)
            logging.info("Outliers capped to interquartile range.")

        except Exception as e:
            logging.error(f"Error during error correction: {e}")
            raise

    def generate_report(self, output_path):
        """Generate a detailed data quality report."""
        logging.info("Generating data quality report...")
        try:
            report_path = output_path + f"/data_quality_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            with open(report_path, 'w') as report_file:
                for key, value in self.report.items():
                    report_file.write(f"{key}: {value}\n")
            logging.info(f"Data quality report saved to {report_path}")
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            raise

    def save_cleaned_data(self, output_path):
        """Save the cleaned and corrected data to a CSV file."""
        logging.info("Saving cleaned data...")
        try:
            cleaned_data_path = output_path + f"/cleaned_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            self.data.to_csv(cleaned_data_path, index=False)
            logging.info(f"Cleaned data saved to {cleaned_data_path}")
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")
            raise

if __name__ == "__main__":
    data_path = "sample_data.csv"
    output_path = "output"

    dqg = DataQualityGovernance(data_path)

    try:
        # Validate data integrity
        dqg.validate_data_integrity()

        # Monitor quality metrics
        dqg.monitor_quality_metrics()

        # Perform error correction
        dqg.error_correction()

        # Generate and save reports
        dqg.generate_report(output_path)
        dqg.save_cleaned_data(output_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

