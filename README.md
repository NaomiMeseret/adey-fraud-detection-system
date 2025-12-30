# Fraud Detection for E-commerce and Bank Transactions 🚨💳

## 📊 Key EDA Visualizations

> Here are some of the main project insights — see the `outputs/eda/` folders for more details.

### Credit Card Fraud Data

- **Class Imbalance**  
  ![Credit Card Class Distribution](outputs/eda/creditcard/creditcard_class_distribution.png)  
  _(Very few transactions are fraudulent! Solving this imbalance is critical.)_ 🟦🟥

- **Transaction Amount by Fraud Class**  
  ![Credit Card Amount by Class](outputs/eda/creditcard/creditcard_amount_by_class.png)  
  _(Fraud amounts can have a different distribution from non-fraud. Look for outliers!)_ 💸

- **Top Feature Correlations**  
  ![Credit Card Correlation Analysis](outputs/eda/creditcard/creditcard_correlation_analysis.png)  
  _(Identify which features most separate fraud from legit transactions.)_ 📈

---

### E-commerce Fraud Data

- **Class Imbalance**  
  ![FraudData Class Distribution](outputs/eda/fraud-data/fraud_class_distribution.png)  
  _(Most transactions are genuine. Imbalance must be handled in modeling.)_ 🟩🟥

- **Fraud by Country**  
  ![Fraud by Country](outputs/eda/fraud-data/fraud_by_country.png)  
  _(Some countries have higher fraud rates. Geolocation patterns can reveal threats.)_ 🌍

- **Fraud by Categorical Feature (e.g., Source/Browser)**  
  ![Fraud Rate by Categorical Features](outputs/eda/fraud-data/fraud_rate_by_categorical_features.png)  
  _(Certain browser, device, or source channels may be riskier!)_ 🕵️‍♂️

---

## 📝 What's this?

This repo helps **detect fraud** in e-commerce and banking using machine learning and exploratory data analysis.

- Handles data imbalance
- Builds & explains predictive models
- Uses geolocation, time, amount, and user patterns to spot fraud
- Provides model explainability using SHAP

---

## 🏗️ Project Structure

```
adey-fraud-detection-system/
├── data/
│   ├── raw/              # Original datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
│   └── processed/        # Cleaned and feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb          # EDA for e-commerce fraud data
│   ├── eda-creditcard.ipynb          # EDA for credit card fraud data
│   ├── feature-engineering.ipynb    # Feature engineering and preprocessing
│   ├── modeling.ipynb                # Model training and evaluation
│   └── shap-explainability.ipynb     # Model explainability with SHAP
├── models/               # Saved trained models (.pkl files)
├── outputs/
│   └── eda/             # Generated visualizations organized by notebook
│       ├── creditcard/
│       ├── fraud-data/
│       ├── feature-engineering/
│       ├── modeling/
│       └── shap/
├── src/                  # Source code modules (if any)
├── tests/                # Unit tests
├── scripts/              # Utility scripts
├── .github/
│   └── workflows/        # CI/CD workflows
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

### Installation Steps

1. **Clone or download the repository**

   ```bash
   git clone <repository-url>
   cd adey-fraud-detection-system
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Jupyter kernel** (if using Jupyter Notebook/Lab)

   ```bash
   python -m ipykernel install --user --name fraud-detection --display-name "Python (fraud-detection)"
   ```

6. **Place your data files** in `data/raw/`:
   - `Fraud_Data.csv` - E-commerce transaction data
   - `creditcard.csv` - Credit card transaction data
   - `IpAddress_to_Country.csv` - IP address to country mapping

---

## 📖 Usage

### Running the Notebooks

1. **Open Jupyter Notebook/Lab**

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Select the kernel**: Choose "Python (fraud-detection)" from the kernel menu

3. **Run notebooks in order**:
   - `notebooks/eda-fraud-data.ipynb` - Explore e-commerce fraud data
   - `notebooks/eda-creditcard.ipynb` - Explore credit card fraud data
   - `notebooks/feature-engineering.ipynb` - Create features and preprocess data
   - `notebooks/modeling.ipynb` - Train and evaluate models
   - `notebooks/shap-explainability.ipynb` - Interpret model predictions

### Expected Outputs

- **Processed data**: Saved to `data/processed/` after feature engineering
- **Trained models**: Saved to `models/` after model training
- **Visualizations**: Saved to `outputs/eda/` organized by notebook

---

## 🔍 Key Features

### Data Analysis

- **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and class distribution analysis
- **Feature Engineering**: Time-based features, transaction velocity, geolocation mapping
- **Data Transformation**: Normalization, encoding, handling missing values

### Model Building

- **Baseline Model**: Logistic Regression with class balancing
- **Ensemble Models**: Random Forest and XGBoost with hyperparameter tuning
- **Cross-Validation**: Stratified K-Fold (k=5) for reliable performance estimation
- **Evaluation Metrics**: AUC-PR, F1-Score, ROC-AUC, Precision, Recall, Confusion Matrix

### Model Explainability

- **SHAP Analysis**: Global and local feature importance
- **Individual Predictions**: Force plots for TP, FP, FN cases
- **Business Recommendations**: Actionable insights based on SHAP analysis

---

## 📊 Evaluation Metrics

The project uses multiple metrics to evaluate model performance:

- **AUC-PR (Average Precision)**: Primary metric for imbalanced data
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision**: Proportion of predicted frauds that are actual frauds
- **Recall**: Proportion of actual frauds that are correctly identified
- **Confusion Matrix**: Detailed breakdown of predictions vs. actuals

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting ensemble
- **imbalanced-learn**: Handling class imbalance (SMOTE)
- **matplotlib & seaborn**: Data visualization
- **SHAP**: Model explainability
- **Jupyter**: Interactive development environment

---

