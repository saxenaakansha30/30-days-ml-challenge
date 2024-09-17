# Problem: Determine Credit Card defaults using a Random Forest Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data.
# Download the data from Default of Credit Card Clients Dataset Kaggle and put it in dataset directory.
data_df = pd.read_csv('dataset/UCI_Credit_Card.csv')
# print(data_df.head())

# Step 2: Preprocess the data.
X = data_df.drop('default.payment.next.month', axis=1) # Features
y = data_df['default.payment.next.month'] # target

# Step 3: Split the data
# Split into 80-20 ratio. Training (80%) and Validation (20%) datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the model
model = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=8, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Prediction and evaluate.
predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
classification_report =  classification_report(y_val, predictions)
print("Classfication Report:\n", classification_report)

# Step 6: Visualization
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()



