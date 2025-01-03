
# Data Quality and Governance Automation

## Overview

The **Data Quality and Governance Automation Framework** is a Python-based solution designed to:

- Automate data quality validation and anomaly detection.
- Monitor key data governance metrics such as completeness, accuracy, and consistency.
- Correct common data issues like missing values, duplicates, and outliers.
- Generate detailed reports and save cleaned datasets for further analysis.

This framework is ideal for organizations looking to enhance their data governance practices and ensure actionable, high-quality data.

---

## Features

1. **Quality Checks**:
   - Automates validation of data integrity, including checks for missing values, duplicates, and outliers.
   - Uses `IsolationForest` for robust anomaly detection.

2. **Governance Metrics**:
   - Monitors and reports on key data quality metrics such as completeness and invalid entries.
   - Provides a comprehensive view of data health.

3. **Error Correction**:
   - Imputes missing values using mean imputation for numeric columns.
   - Removes duplicate rows.
   - Caps outliers to the interquartile range for improved consistency.

4. **Reporting**:
   - Generates a detailed data quality report with key findings.
   - Saves cleaned and corrected data to a CSV file with a timestamped filename.

5. **Logging**:
   - Logs all operations and outcomes, ensuring transparency and facilitating debugging.

---

## Installation

### Prerequisites

- Python 3.7+
- Required Libraries:
  - pandas
  - numpy
  - scikit-learn
  - logging

### Setup

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Provide a CSV dataset (`sample_data.csv`) in the root directory. Ensure the dataset contains the following:
   - Numeric and categorical columns for validation.
   - Any specific domain-relevant data for advanced validation (e.g., dates).

---

## Usage

### Running the Framework

To execute the framework, run the main script:
```bash
python data_quality_governance.py
```

### Key Steps Executed:

1. **Validation**:
   - Identifies missing values, duplicates, and anomalies in the dataset.

2. **Monitoring**:
   - Reports on data completeness and other key metrics.

3. **Error Correction**:
   - Imputes missing values, removes duplicates, and caps outliers.

4. **Reporting**:
   - Saves a detailed data quality report as a text file.

5. **Output**:
   - Saves the cleaned dataset as a CSV file with a timestamped filename.

---

## File Structure

```
Data_Quality_Governance/
├── data_quality_governance.py       # Main framework script
├── requirements.txt                # Dependencies
├── sample_data.csv                 # Sample dataset (user-provided)
├── README.md                       # Documentation
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
