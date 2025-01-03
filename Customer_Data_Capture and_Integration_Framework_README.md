
# Customer Data Capture and Integration Framework

## Overview

The **Customer Data Capture and Integration Framework** is a Python-based solution designed to:

- Extract customer data from various sources, including APIs, databases, and CSV files.
- Clean and transform the data to ensure consistency and quality.
- Integrate datasets into a unified view to provide actionable insights.
- Save the processed data into a centralized database.
- Generate summary statistics for further analysis.

This framework is modular, scalable, and adheres to professional standards, making it suitable for use in business intelligence, customer analysis, and data engineering projects.

---

## Features

1. **Data Extraction**:
   - Fetches data from APIs using RESTful endpoints.
   - Queries relational databases via SQLAlchemy.
   - Reads data from local CSV files.

2. **Data Cleaning**:
   - Handles missing values and duplicates.
   - Standardizes inconsistent formats (e.g., email, phone numbers).
   - Ensures accurate date-time parsing.

3. **Data Integration**:
   - Merges datasets to create a unified view of customer behavior.
   - Removes duplicate records based on unique customer identifiers.

4. **Output**:
   - Provides clean, integrated datasets.
   - Generates summary statistics for quick analysis.
   - Saves processed data into a centralized database table.

5. **Error Handling**:
   - Comprehensive error handling for API calls, database queries, and file operations.

6. **Unit Testing**:
   - Includes unit tests for key components to ensure reliability and correctness.

---

## Installation

### Prerequisites

- Python 3.7+
- Libraries:
  - pandas
  - requests
  - SQLAlchemy
  - unittest (for testing)

### Setup
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the framework:
   - Update the `API_URL`, `DATABASE_CONNECTION_STRING`, and `CSV_FILE_PATH` in the script to match your data sources.

---

## Usage

### Running the Framework

To execute the framework, run the main script:
```bash
python customer_data_framework.py
```

This will:
- Extract data from the specified sources.
- Clean, transform, and integrate the data.
- Save the processed data into the database.
- Generate and display summary statistics.

### Running Unit Tests

To validate the framework, execute the test suite:
```bash
python -m unittest customer_data_framework_tests.py
```

---

## File Structure

```
Customer_Data_Framework/
├── customer_data_framework.py       # Main framework script
├── customer_data_framework_tests.py # Unit tests for the framework
├── data/                            # Directory for sample CSV files
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## Example Output

### Summary Statistics:
```text
Summary Statistics:
Total records: 1000
Columns:
- email
- phone
- address
- created_at
```

### Log Output:
```text
2022-01-03 12:00:00 - INFO - Fetching data from API...
2022-01-03 12:00:05 - INFO - Data fetched successfully from API.
2022-01-03 12:00:10 - INFO - Cleaning and transforming data...
2022-01-03 12:00:15 - INFO - Data integrated successfully.
2022-01-03 12:00:20 - INFO - Data saved to database successfully.
```

---

## Contributing

Contributions are welcome! 

---

## License

This project is licensed under the MIT License. 

---

## Contact

For questions or feedback, please contact:
- **Email**: johnjohnsonogbidi@gmail.com
