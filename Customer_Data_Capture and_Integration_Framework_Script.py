#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
from sqlalchemy import create_engine
import logging
import unittest
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
API_URL = "https://example.com/api/customers"
DATABASE_CONNECTION_STRING = "mysql+pymysql://username:password@host:port/database"
CSV_FILE_PATH = "data/customers.csv"
TABLE_NAME = "centralized_customers"

class CustomerDataFramework:
    def __init__(self, api_url, db_connection_string, csv_file_path):
        self.api_url = api_url
        self.db_connection_string = db_connection_string
        self.csv_file_path = csv_file_path

    def fetch_data_from_api(self):
        """Fetch customer data from an API."""
        try:
            logging.info("Fetching data from API...")
            response = requests.get(self.api_url)
            response.raise_for_status()
            logging.info("Data fetched successfully from API.")
            return pd.DataFrame(response.json())
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from API: {e}")
            return pd.DataFrame()

    def fetch_data_from_database(self, query):
        """Fetch customer data from a database."""
        try:
            logging.info("Fetching data from database...")
            engine = create_engine(self.db_connection_string)
            with engine.connect() as conn:
                data = pd.read_sql(query, conn)
            logging.info("Data fetched successfully from database.")
            return data
        except Exception as e:
            logging.error(f"Error fetching data from database: {e}")
            return pd.DataFrame()

    def fetch_data_from_csv(self):
        """Fetch customer data from a CSV file."""
        try:
            logging.info("Reading data from CSV file...")
            data = pd.read_csv(self.csv_file_path)
            logging.info("Data read successfully from CSV file.")
            return data
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()

    def clean_and_transform_data(self, df):
        """Clean and transform customer data."""
        try:
            logging.info("Cleaning and transforming data...")
            # Handle missing values
            df = df.dropna(subset=["email", "phone"])
            df.fillna({"address": "Unknown", "created_at": "1970-01-01"}, inplace=True)

            # Standardize formats
            df["email"] = df["email"].str.lower()
            df["phone"] = df["phone"].str.replace("[^0-9]", "", regex=True)
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

            # Remove duplicates
            df = df.drop_duplicates(subset=["email"])
            logging.info("Data cleaned and transformed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return df

    def integrate_data(self, combined_data):
        """Integrate datasets into a unified view."""
        try:
            logging.info("Integrating data...")
            unified_data = combined_data.groupby("email").first().reset_index()
            logging.info("Data integrated successfully.")
            return unified_data
        except Exception as e:
            logging.error(f"Error during data integration: {e}")
            return combined_data

    def generate_summary_statistics(self, df):
        """Generate summary statistics for the dataset."""
        try:
            logging.info("Generating summary statistics...")
            summary = df.describe(include="all")
            logging.info("Summary statistics generated successfully.")
            logging.info(f"Total records: {len(df)}")
            logging.info(f"Summary:\n{summary}")
        except Exception as e:
            logging.error(f"Error generating summary statistics: {e}")

    def save_to_database(self, df):
        """Save the cleaned and integrated data to a database."""
        try:
            logging.info("Saving data to database...")
            engine = create_engine(self.db_connection_string)
            with engine.connect() as conn:
                df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            logging.info("Data saved to database successfully.")
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")

    def run(self):
        """Execute the complete framework."""
        logging.info("Starting Customer Data Capture and Integration Framework...")
        api_data = self.fetch_data_from_api()
        db_data = self.fetch_data_from_database("SELECT * FROM customers")
        csv_data = self.fetch_data_from_csv()

        combined_data = pd.concat([api_data, db_data, csv_data], ignore_index=True)
        cleaned_data = self.clean_and_transform_data(combined_data)
        unified_data = self.integrate_data(cleaned_data)
        self.generate_summary_statistics(unified_data)
        self.save_to_database(unified_data)
        logging.info("Framework execution completed successfully.")

class TestCustomerDataFramework(unittest.TestCase):

    @patch("requests.get")
    def test_fetch_data_from_api(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [{"email": "test@example.com", "phone": "1234567890"}]
        framework = CustomerDataFramework(API_URL, DATABASE_CONNECTION_STRING, CSV_FILE_PATH)
        result = framework.fetch_data_from_api()
        self.assertEqual(len(result), 1)
        self.assertIn("email", result.columns)

    @patch("sqlalchemy.create_engine")
    def test_fetch_data_from_database(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [("test@example.com", "1234567890")]
        framework = CustomerDataFramework(API_URL, DATABASE_CONNECTION_STRING, CSV_FILE_PATH)
        result = framework.fetch_data_from_database("SELECT * FROM customers")
        self.assertIsInstance(result, pd.DataFrame)

    @patch("pandas.read_csv")
    def test_fetch_data_from_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({"email": ["test@example.com"], "phone": ["1234567890"]})
        framework = CustomerDataFramework(API_URL, DATABASE_CONNECTION_STRING, CSV_FILE_PATH)
        result = framework.fetch_data_from_csv()
        self.assertEqual(len(result), 1)

    def test_clean_and_transform_data(self):
        data = pd.DataFrame({
            "email": ["TEST@EXAMPLE.COM", "test@example.com"],
            "phone": ["123-456-7890", None],
            "created_at": ["2022-01-01", None]
        })
        framework = CustomerDataFramework(API_URL, DATABASE_CONNECTION_STRING, CSV_FILE_PATH)
        result = framework.clean_and_transform_data(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["email"].iloc[0], "test@example.com")

    def test_integrate_data(self):
        data = pd.DataFrame({
            "email": ["test@example.com", "test@example.com"],
            "phone": ["1234567890", "1234567890"]
        })
        framework = CustomerDataFramework(API_URL, DATABASE_CONNECTION_STRING, CSV_FILE_PATH)
        result = framework.integrate_data(data)
        self.assertEqual(len(result), 1)

if __name__ == "__main__":
    unittest.main()

