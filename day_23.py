# Problem: Fraud Detection in Financial Transactions using Logistic Regression and Random Forest
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
# The dataset is highly unbalanced.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the datasets
data = pd.read_csv('dataset/creditcard.csv')
# print(data.info())
# print(data.head())

# Step 2: Data Preprocessing
# Handle missing values.
# print(data.isnull().sum())
# We dont have any column with null value.

# Create feature and target datasets.
X = data.drop('Class', axis=1) # Features
y = data['Class'] # Target, 1 in case of fraud and 0 otherwise.

# Split the dataset into training (80%) and validation (20%) datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 3: Train and Evaluate the Logistic Regression model.
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)

# Make Prediction and Evaluate.
log_reg_predictions = log_reg_model.predict(X_val_scaled)

# Evaluation of model's performance.
log_reg_accuracy = accuracy_score(y_val, log_reg_predictions)
log_reg_conf_matrix = confusion_matrix(y_val, log_reg_predictions)
log_reg_classfication_report = classification_report(y_val, log_reg_predictions)

print(f"Accuracy Score of Logistic Regression\n: {log_reg_accuracy}")
print(f"Confusion Matrix of Logistic Regression\n: {log_reg_conf_matrix}")
print(f"Classfication Report of Logistic Regression\n: {log_reg_classfication_report}")

# Step 4: Train and Evaluate the Random Forest Model.
ran_for_model = RandomForestClassifier(random_state=42)
ran_for_model.fit(X_train, y_train)

# Make prediction
ran_for_predictions = ran_for_model.predict(X_val)

# Evaluate the model's performance.
ran_for_accuracy = accuracy_score(y_val, ran_for_predictions)
ran_for_conf_matrix = confusion_matrix(y_val, ran_for_predictions)
ran_for_classfication_report = classification_report(y_val, ran_for_predictions)

print(f"Accuracy Score of Random Forest\n: {ran_for_accuracy}")
print(f"Confusion Matrix of Random Forest\n: {ran_for_conf_matrix}")
print(f"Classfication Report of Random Forest\n: {ran_for_classfication_report}")

# Step 5: Visualization
# Logistic Regression
plt.figure(figsize=(7, 5))
sns.heatmap(log_reg_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Random Forest
plt.figure(figsize=(7, 5))
sns.heatmap(ran_for_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Random Forest')
plt.show()


# Output:
# Accuracy Score of Logistic Regression
# : 0.9991222218320986
# Confusion Matrix of Logistic Regression
# : [[56855     9]
#  [   41    57]]
# Classfication Report of Logistic Regression
# :               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00     56864
#            1       0.86      0.58      0.70        98
#
#     accuracy                           1.00     56962
#    macro avg       0.93      0.79      0.85     56962
# weighted avg       1.00      1.00      1.00     56962
#
# Accuracy Score of Random Forest
# : 0.9995611109160493
# Confusion Matrix of Random Forest
# : [[56862     2]
#  [   23    75]]
# Classfication Report of Random Forest
# :               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00     56864
#            1       0.97      0.77      0.86        98
#
#     accuracy                           1.00     56962
#    macro avg       0.99      0.88      0.93     56962
# weighted avg       1.00      1.00      1.00     56962