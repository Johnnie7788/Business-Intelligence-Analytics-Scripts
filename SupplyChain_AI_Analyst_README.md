
# SupplyChain AI Analyst

## Overview

The SupplyChain AI Analyst Script is a machine learning-powered tool designed to help businesses analyze, predict, and optimize supply chain performance. This script processes key supply chain data—such as inventory levels, lead times, supplier ratings, and units sold—using techniques like Random Forest for predictive analysis and Prophet for time-series forecasting. It provides detailed insights and recommendations, enabling management to make data-driven decisions that improve overall supply chain efficiency.

## Cloud Integration

This script can be easily integrated with cloud platforms like AWS, Azure, or Google Cloud for data storage, scalability, and the use of cloud-based AI/ML services. Cloud services like AWS S3, Azure Blob, or Google Cloud Storage can manage large datasets, while services like AWS SageMaker, Azure Machine Learning, or Google AI can be used for model training, deployment, and real-time predictions.

## Key Features Include:

- **Inventory Optimization:** Uses Random Forest Regressor to optimize inventory levels based on lead times, supplier performance, and pricing factors.
- **Supplier Performance Clustering:** Groups suppliers into clusters based on their performance, identifying top-performing suppliers and highlighting areas for improvement.
- **Demand Forecasting:** Implements Prophet for time-series forecasting, predicting future demand trends to guide decision-making and inventory management.
- **Detailed Visualizations:** Generates visualizations of supplier performance, demand forecasts, and inventory trends, making it easier to identify patterns.
- **Clustering Analysis:** Groups suppliers into performance clusters to discover best practices and encourage optimization across the supply chain.
- **Professional Summary:** Provides a summary of supply chain performance with actionable recommendations for improving efficiency and reducing risk.

## Key Features & Functionality

### 1. Data Loading and Error Handling

The script loads the dataset from a CSV file and automatically handles missing values by filling them with the median for numerical columns. This ensures that the dataset is clean and ready for analysis.

### 2. Inventory Optimization

The script uses a Random Forest Regressor to optimize the inventory levels based on:

- Inventory Level
- Lead Time (Days)
- Supplier Rating (Scale of 1-5)
- Price per Unit (EUR)
- Location
- Customer Type

The model provides predictions that can help management optimize stock levels and avoid overstocking or stockouts.

### 3. Supplier Performance Clustering

The script automatically calculates supplier performance metrics, such as:

- **Supplier Rating:** Performance score based on product quality and delivery time.
- **Defective Rate:** Percentage of defective items.
- **Lead Time:** Time taken by a supplier to fulfill an order.

### 4. Clustering for Best Practices

The script applies clustering (K-Means) to group suppliers based on their performance and delivery times, helping management identify top-performing suppliers and areas for improvement.

### 5. Demand Forecasting

The script utilizes Prophet for time-series forecasting to predict future demand trends over the next 90-120 days. This helps businesses anticipate changes in demand and adjust their inventory strategies accordingly.

### 6. Visualizations

The script produces several key visualizations, such as:

- **Supplier Performance Clusters:** Shows the grouping of suppliers based on their ratings and defect rates.
- **Demand Forecast:** Predicts demand trends over time, helping management plan ahead.
- **Inventory Trends:** Visualizes inventory performance and the impact of various factors on stock levels.

### 7. Automatic Interpretation & Recommendations

For each analysis (e.g., inventory optimization, supplier performance clustering, forecasting), the script automatically generates detailed interpretations and recommendations. This helps to understand current performance and provides actionable strategies for improvement.

### 8. Professional Summary & Next Steps

The script concludes with a high-level professional summary that includes:

- **Supply Chain Performance Overview:** A summary of supplier efficiency, inventory optimization, and demand forecasting.
- **Recommendations:** Actionable suggestions for improving supply chain performance, reducing risks, and enhancing supplier relationships.

## Dataset

The dataset used in this analysis includes the following columns:

- **Date:** The date of the sales transaction.
- **Supplier:** Name of the supplier.
- **Product:** The category or type of product.
- **Units Sold:** The number of units sold during the time period.
- **Inventory Level:** The level of inventory in stock.
- **Lead Time (Days):** The time it takes for the supplier to deliver goods.
- **Supplier Rating (Scale of 1-5):** A rating for supplier performance.
- **Defective Rate (%):** The percentage of defective products received from the supplier.
- **Customer Type:** The type of customer (e.g., business or individual).
- **Location:** The geographical location of the supplier or customer.
- **Price per Unit (EUR):** The price per unit of the product sold.
- **Revenue (EUR):** The revenue generated from the sale of the product.

These columns help analyze supplier performance, inventory optimization, and demand forecasting, making it easier to identify trends and optimize the supply chain.

## How to Use

### 1. Install Required Dependencies

Ensure you have the following Python libraries installed:
```bash
pip install pandas matplotlib seaborn scikit-learn prophet imbalanced-learn
```

### 2. Prepare Your Dataset

Place your supply chain dataset in the root directory with the name `supply_chain_data.csv`.

### 3. Run the Script

Execute the script to:

- **Optimize Inventory Levels:** Use the Random Forest Regressor to optimize stock levels and reduce excess inventory.
- **Analyze Supplier Performance:** Automatically calculate supplier performance, identify top performers, and provide strategies for improving supplier relationships.
- **Forecast Demand Trends:** Use Prophet to predict future demand trends over the next 90-120 days and adjust inventory accordingly.

### 4. Review the Outputs

The script will generate:

- **Supplier Performance Clusters:** Visual representation of supplier performance and defect rates.
- **Inventory Optimization Metrics:** Predictive metrics on optimal stock levels.
- **Demand Forecast:** A time-series forecast of demand for the next 90-120 days.

## Libraries Used

- **Pandas:** For data manipulation and analysis.
- **Matplotlib & Seaborn:** For generating visualizations.
- **scikit-learn:** For machine learning models and clustering.
- **Prophet:** For time-series forecasting.
- **imbalanced-learn (SMOTE):** For handling data imbalance in the dataset.

## Contribution

Contributions are welcome! 

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this script as long as proper credit is given.
