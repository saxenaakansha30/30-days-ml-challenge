# Day 6 problem: Predict wine quality from physicochemical properties using SVM (Support Vector Machines)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data.
data_df = pd.read_csv('dataset/WineQT.csv', sep=',')

# Step 2: Create feature and target dataset
X = data_df.drop('quality', axis=1)
y = data_df['quality']

# Step 3: Data Preprocessing.
# Scaling transformation is the process of transforming your data so that all features (variables) are on a similar scale or range.
# It’s commonly done in machine learning to ensure that no feature dominates the others simply because of its larger numerical range.

# StandardScaler ensures that all features contribute equally by transforming the data to have a mean of 0 and a standard deviation of 1.
standard_scale = StandardScaler()
X_scaled = standard_scale.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
#print(X_scaled_df.head())

# Step 4: Split the data.
# Split into 80-20 ratio. Training (80%) and validation(20%)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Create and Train the Model.
# SVC is a classifier type model of SVM.
# This model creates a hyperplane to separate two classes. Like a good quality wine and poor quality wine.
# Visualize the hyperplane like a straight line in 2D can visualize it like a plane in 3D.
# If the dataset can be easily separatable, we use linear type kernal to mention straight boundary.
# If the dataset is complex and can not be easily separatable, we use rbf type kernal to mention a curved decision boundary.
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Step 6: Make predictions and Evaluate
predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy Score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
# For class imbalance: If we do not have enough data of class (quality=3) in predicted values.
# We can tell classification report to set the precision and F-score to 0 when there are no predicted samples for a class.
classification_report = classification_report(y_val, predictions, zero_division=0)
print("Classfication Report:\n", classification_report)


# Step 7: Visualization
plt.figure(figsize=(8,7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Confusion Matrix')
plt.show()
