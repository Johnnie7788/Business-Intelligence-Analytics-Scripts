
# Customer Propensity and Motivational Analysis

## Overview

The **Customer Propensity and Motivational Analysis Framework** is a Python-based solution designed to:

- Predict customer purchase likelihood using machine learning models like Random Forest and XGBoost.
- Perform sentiment analysis on customer reviews and feedback to understand overall sentiment.
- Analyze customer motivations and trends using Natural Language Processing (NLP) techniques like word clouds and topic modeling.
- Visualize feature importance to identify key drivers of customer behavior.

This framework combines data science, machine learning, and NLP to support business decision-making and improve customer engagement strategies.

---

## Features

1. **Predictive Modeling**:
   - Implements Random Forest and XGBoost for purchase propensity prediction.
   - Generates classification reports for model evaluation.

2. **Sentiment Analysis**:
   - Analyzes customer feedback using NLTK's Sentiment Intensity Analyzer.
   - Visualizes sentiment distribution with histograms.

3. **Motivational Trend Analysis**:
   - Creates word clouds to represent customer motivations visually.
   - Uses Latent Dirichlet Allocation (LDA) for topic modeling to uncover dominant themes in customer feedback.

4. **Feature Importance Visualization**:
   - Highlights the most influential features in predicting customer purchase likelihood.

5. **Error Handling and Logging**:
   - Includes robust error handling to manage runtime issues.
   - Provides detailed logging for debugging and monitoring.

---

## Installation

### Prerequisites

- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - wordcloud
  - nltk

### Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download necessary NLTK resources:
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

3. Provide a CSV dataset named `customer_data.csv` in the root directory. The dataset should include the following columns:
   - `age`
   - `income`
   - `purchase` (target variable)
   - `customer_reviews` (for sentiment analysis and NLP tasks)

---

## Usage

### Running the Framework

To execute the framework, run the main script:
```bash
python customer_propensity_analysis.py
```

### Expected Output

1. **Predictive Modeling**:
   - Classification reports for Random Forest and XGBoost.

2. **Sentiment Analysis**:
   - Histogram of sentiment scores.

3. **Motivational Trend Analysis**:
   - Word cloud of customer motivations.
   - Top words for each topic discovered through LDA.

4. **Feature Importance Visualization**:
   - Horizontal bar chart showing feature importance.

## File Structure

```
Customer_Propensity_Analysis/
├── customer_propensity_analysis.py  # Main framework script
├── requirements.txt                 # Dependencies
├── customer_data.csv                # Sample dataset (user-provided)
├── README.md                        # Documentation
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
