# Problem: Customer churn prediction with XGBoost
# Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data
data = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# print(data.info()) # Let's explore the data.

# Step 2: Preprocess the data
# Handle missing values.
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(data['TotalCharges'].median(), inplace=True)
# We handle missing values in the TotalCharges column by converting it to numeric and filling
# the missing values with the median.

data.drop('customerID', axis=1, inplace=True)

# Extract the target 'Churn' before encoding.
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert 'Yes' to 1 and 'No' to 0
# Drop churn from the original dataset to evoid getting it encoded.
data.drop('Churn', axis=1)

# Convert binary categorical data to numeric using Label Encoding
binary_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

# Apply Label encoding on Binary Categorical Columns.
label_encoder = LabelEncoder()
for col in binary_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Define multi-value categorical columns list to encode using One Hot Encoding.
multi_class_col = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']

# Apply one hot encoding on multi class columns using ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(drop='first'), multi_class_col)], remainder='passthrough')
data_encoded = pd.DataFrame(ct.fit_transform(data))

# Step 3: Define Feature and Target Dataset
X = data_encoded

# Step 4: Split data into training and validation dataset.
# We will go with classic 80-20 ratio.
# 80% -  Training data
# 20% - Validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the model.
model = XGBClassifier(n_estimators=100, learning_rate=1.1, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make prediction and evaluate.
predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
confusion_matrix = confusion_matrix(y_val, predictions)
classification_report = classification_report(y_val, predictions)

print(f"Accuracy Score: {accuracy_score}")
print(f"Confusion Matrix: {confusion_matrix}")
print(f"Classification Report: {classification_report}")


# Step 7: Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Churn Predicted')
plt.show()

# Output:

# Accuracy Score: 1.0
# Confusion Matrix: [[1036    0]
#  [   0  373]]
# Classification Report:               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00      1036
#            1       1.00      1.00      1.00       373
#
#     accuracy                           1.00      1409
#    macro avg       1.00      1.00      1.00      1409
# weighted avg       1.00      1.00      1.00      1409



