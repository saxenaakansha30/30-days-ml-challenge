# Problem: Capstone Project: Predicting loan approvals using ensemble learning (Random Forest, XGBoost)
# Dataset: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data.
data = pd.read_csv('dataset/loan_approval_dataset.csv')
# print(data.info())
# print(data.head())

# Step 2: Preprocess the data
# Check for missing values
#print(data.isnull().sum()) # We do not have any missing values.

# Find the categorical data.
# print(data[' education'].unique()) # [' Graduate' ' Not Graduate']
# print(data[' self_employed'].unique()) # [' No' ' Yes']
# print(data[' loan_status'].unique()) # [' Approved' ' Rejected']

# The dataset has an extra space before the actual value.
# Update the column name first.
data.columns = data.columns.str.strip()

# Now the values.
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()

# Convert the categorical data into numeric using One-Hot Encoding
data = pd.get_dummies(data, columns=['education', 'self_employed'], drop_first=False)

# For consistency, model expects in 0, 1. Change Rejected as 0 and Approved as 1
data['loan_status'] = data['loan_status'].replace({'Approved': 1, 'Rejected': 0}) # [1 0]

# Create features and Target Dataset
X = data.drop('loan_status', axis=1)  # Feature Dataset
y = data['loan_status']  # Target Dataset

# Step 3: Split the datasets into training and validation datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Models

# Random Forest Model
# How it works: It creates multiple decision trees on different subsets of the data, and each tree gives a prediction.
# The final prediction is based on the majority vote (classification) or the average (regression) of all the trees.
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# XGBoost (Extreme Gradient Boosting):
# How it works: XGBoost adds trees one by one, with each tree attempting to minimize the errors made by the previous
# trees using a gradient descent approach. It uses boosting techniques to improve performance.
model_xgb = XGBClassifier(n_estimators=100, learning_rate=1.1, random_state=42)
model_xgb.fit(X_train, y_train)

# Step 5: Make Predictions and Evaluate
# Random Forest
predictions_rf = model_rf.predict(X_val)

accuracy_score_rf = accuracy_score(predictions_rf, y_val)
confusion_matrix_rf = confusion_matrix(predictions_rf, y_val)
classification_report_rf = classification_report(predictions_rf, y_val)

print("Random Forest: ")
print(f"Accuracy Score: {accuracy_score_rf}")
print(f"Confusion Matrix: {confusion_matrix_rf}")
print(f"Classification Report: {classification_report_rf}")

# XGBoost
predictions_xgb = model_xgb.predict(X_val)

accuracy_score_xbg = accuracy_score(predictions_xgb, y_val)
confusion_matrix_xgb = confusion_matrix(predictions_xgb, y_val)
classification_report_xgb = classification_report(predictions_xgb, y_val)

print("XGBoost: ")
print(f"Accuracy Score: {accuracy_score_xbg}")
print(f"Confusion Matrix: {confusion_matrix_xgb}")
print(f"Classification Report: {classification_report_xgb}")

# Step 6: Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Random Forest Confusion Matrix')


plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('XGBoost Confusion Matrix')

plt.show()