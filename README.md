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

## 📝 What’s this?
This repo helps **detect fraud** in e-commerce and banking using machine learning and exploratory data analysis.  
- Handles data imbalance  
- Builds & explains predictive models  
- Uses geolocation, time, amount, and user patterns to spot fraud

---

## 🚀 Instructions

1. 📂 Place data in `data/raw/`
2. ▶️ Run notebooks in order (EDA → Feature Engineering → Modeling)
3. 🖼️ See all saved plots in `outputs/eda/creditcard/` and `outputs/eda/fraud-data/`

---

