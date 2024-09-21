#Problem: Snomaly detection with Isolation Forest in Credit Card Fraud Detection dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data
data = pd.read_csv('dataset/creditcard.csv')

# Step 2: Create Feature and Target datasets.
X = data.drop('Class', axis=1) # Feature
y = data['Class'] # Target, 0 for normal transaction and 1 for fraud.
y = y.map({0: 1, 1: -1}) # Change for consistency: -1, means anomaly, 1 means normal)

# print(y.value_counts())
# print(X.head())
# print(y.head())

# Step 3: Split the data: into training and validation datasets.
# We split into 80-20 ratio. Training (80) and validation (20%)
# stratify=y: This ensures the class distribution (fraud vs. non-fraud) remains consistent in both the training and testing sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#print(X_train.shape, X_val.shape)

# Step 4: Build and train the model
model = IsolationForest(contamination=0.01, random_state=42)
# Contamination refers to the proportion of anomalies (outliers) in your dataset. Essentially,
# it’s the expected fraction of the dataset that is considered to be anomalies.
# random_state controls the randomness of the model. It ensures that if you run the algorithm multiple times
# with the same data, you get the same results each time.
model.fit(X_train)

# Step 5: Make predictions and Evaluate
# Predict anomalies on test data, (-1, means anomaly, 1 means normal)
predictions = model.predict(X_val)
X_val['anomaly'] = predictions
#print(predictions)

accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy Score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions, zero_division=1)
print("Classfication Report:\n", classification_report)

# Step 6: Visualization
# Visualize the anomalies vs normal points using two features (e.g., 'V1' and 'V2')
plt.figure(figsize=(10, 6))
plt.scatter(X_val['V1'], X_val['V2'], c=predictions, cmap='coolwarm', label='Anomalies')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Isolation Forest: Anomalies vs Normal Transactions')
plt.legend()
plt.show()

# Visualize the True Positives, False Positives, True Negatives and False Negatives using confusion matrix.
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['True Anomaly', 'True Normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Matrix Label Breakdown:
# Top-left (True Fraud predicted as Fraud): 53
# Top-right (True Fraud predicted as Normal): 45
# Bottom-left (True Normal predicted as Fraud): 557
# Bottom-right (True Normal predicted as Normal): 56307