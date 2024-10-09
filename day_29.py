# Problem: 	Credit risk prediction with Logistic Regression and SVM
# Dataset: https://www.kaggle.com/datasets/uciml/german-credit

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Load the data
data = pd.read_csv('dataset/german_credit_data.csv')

# Step 2: Preprocess the data
# Remove the first column as it is unnamed in the csv file.
data = data.iloc[:, 1:]

# print(data.isnull().sum())  # Check the count of missing values in dataset
# Saving accounts has 183 missing values.
# Checking account has 394 missing values.

data['Saving accounts'].fillna('unknown', inplace=True)
data['Checking account'].fillna('unknown', inplace=True)

# Convert the categorical columns into numeric using One-Hot Encoding
data = pd.get_dummies(data, columns=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'], drop_first=True)

# Step 3: Create Feature and Target datasets
# Define a simple rule of generating risk column
# If the account has credit amount of 5000 and the Duration is more than 24 hours, it is considered a high risk.
data['risk'] = ((data['Credit amount'] > 5000) & (data['Duration'] > 24)).astype(int)

X = data.drop('risk', axis=1) # Features
y = data['risk'] # Target

# Step 4: Split the data into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 6: Build and Train the Logistic Regression Model
model_log_reg = LogisticRegression()
model_log_reg.fit(X_train_scaled, y_train)

# Step 7: Build and Train the SVM Model
model_svm = SVC()
model_svm.fit(X_train_scaled, y_train)

# Step 8: Make Prediction and Evaluate Logistic Regression Model
log_reg_predictions = model_log_reg.predict(X_val_scaled)
accuracy_score_lg = accuracy_score(log_reg_predictions, y_val)
confusion_matrix_lg = confusion_matrix(log_reg_predictions, y_val)
classification_report_log_reg = classification_report(log_reg_predictions, y_val)

print("Logistic Regression: ")
print(f"Accuracy Score: {accuracy_score_lg}")
print(f"Confusion Matrix: \n {confusion_matrix_lg}")
print(f"Classification Report: \n {classification_report_log_reg}")

# Step 9: Make Prediction and Evaluate the SVM Model
svm_predictions = model_svm.predict(X_val_scaled)
accuracy_score_svm = accuracy_score(svm_predictions, y_val)
confusion_matrix_svm = confusion_matrix(svm_predictions, y_val)
classification_report_svm = classification_report(svm_predictions, y_val)

print("SVM Model: ")
print(f"Accuracy Score: {accuracy_score_svm}")
print(f"Confusion Matrix: \n {confusion_matrix_svm}")
print(f"Classification Report: \n {classification_report_svm}")
