# 💳 Credit Card Fraud Detection using Machine Learning

This project uses a **Random Forest Classifier** to detect fraudulent credit card transactions based on a dataset of anonymized transactions.

---

## 🔍 Dataset

- Total Transactions: **284,807**
- Target Variable:
  - `Class = 1` → Fraud
  - `Class = 0` → Valid
- Features:
  - `V1` to `V28` are anonymized using **PCA (Principal Component Analysis)**

---

## 📈 Model

- Algorithm Used: **Random Forest Classifier**
- Accuracy: **~99.95%**
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## 📂 Project Files

- `credit_card_fraud_detection.py` → Main Python implementation  
- `credit.csv` → Dataset  
- `README.md` → Project documentation  

---

## 🛠️ Future Work

- Apply **SMOTE** to handle class imbalance  
- Experiment with other models:
  - XGBoost  
  - Support Vector Machine (SVM)  
  - Logistic Regression  
- Add:
  - ROC Curve  
  - Precision-Recall Curve  

---

## 📌 Requirements

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
