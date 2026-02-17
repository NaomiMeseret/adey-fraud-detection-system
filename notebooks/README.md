# Notebooks

This directory contains Jupyter notebooks for the fraud detection project.

## Notebook Order

1. **data-overview.ipynb**: Data overview 
   - Load both e-commerce and credit card fraud datasets using `src` constants
   - Report shape, columns, class distribution
   - Save overview plot to outputs/eda/overview/

2. **eda-fraud-data.ipynb**: Exploratory Data Analysis for e-commerce transaction data

   - Data cleaning and preprocessing
   - Univariate and bivariate analysis
   - Class distribution analysis
   - Geolocation integration

3. **eda-creditcard.ipynb**: Exploratory Data Analysis for bank credit card transaction data

   - Data exploration
   - Feature analysis
   - Class imbalance analysis

4. **feature-engineering.ipynb**: Feature engineering and data transformation

   - Transaction frequency and velocity features
   - Time-based features
   - IP address to country mapping
   - Data normalization and encoding

5. **modeling.ipynb**: Model training and evaluation

   - Model selection
   - Training with class imbalance handling
   - Model evaluation and comparison
   - Model selection justification

6. **shap-explainability.ipynb**: Model explainability using SHAP
   - SHAP value calculation
   - Feature importance visualization
   - Model decision interpretation

