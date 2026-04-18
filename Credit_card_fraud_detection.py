# 📦 Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# 📥 Load the dataset 
data = pd.read_csv("creditcard_1.csv")  

# 👀 View first few rows
print("📄 First few rows of the dataset:")
print(data.head())

# 📊 Basic info
print("\n🔍 Dataset shape:", data.shape)
print("\n📊 Dataset statistics:\n", data.describe())

# 🏷️ Count of fraud and valid transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print("\n⚠️ Fraudulent Transaction Percentage:", outlier_fraction * 100, "%")
print("✅ Valid Transactions:", len(valid))
print("❌ Fraudulent Transactions:", len(fraud))

# 💰 Analyze amount differences
print("\n💰 Amount details of fraudulent transactions:")
print(fraud['Amount'].describe())

print("\n💰 Amount details of valid transactions:")
print(valid['Amount'].describe())

# 📈 Correlation matrix
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title("📊 Correlation Matrix")
plt.show()

# 🎯 Splitting features and target variable
X = data.drop(['Class'], axis=1)  # Input features
Y = data['Class']                 # Output (fraud/not fraud)
xData = X.values
yData = Y.values

# 🧪 Split data into training and testing
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# 🌳 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

print("\n🚀 Training Random Forest Classifier...")
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# 🎯 Prediction
yPred = rfc.predict(xTest)

# ✅ Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

print("\n🧪 Model Evaluation (Random Forest):")
acc = accuracy_score(yTest, yPred)
prec = precision_score(yTest, yPred)
rec = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print(f"🔹 Accuracy: {acc}")
print(f"🔹 Precision: {prec}")
print(f"🔹 Recall: {rec}")
print(f"🔹 F1 Score: {f1}")
print(f"🔹 Matthews Correlation Coefficient: {mcc}")

# 📉 Confusion Matrix Visualization
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
plt.title("📊 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
