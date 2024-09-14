# The Problem is: Diagnose breast cancer as malignant or benign using a Decision Tree
# It is 3rd classication problem in a series.
# But this day we are going to use a different Model called Decision Tree.
# Let's start as now we have some idea of the Model.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the dataset.
data = load_breast_cancer()

# Step 2: Create feature and target set.
# The data.frame returns None. So have to create Panda Dataframes explicitly
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 3: Split the dataset into training and validation set.
# Like always we will go training (80%) and validation (20%)
# By now, you would already know about the reason for random_state=42
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Prediction and Evaluate
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy is:\n", accuracy)
confusion_matrix = confusion_matrix(y_val, predictions)
print("\nConfusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions)
print("\n Classfication Report:\n", classification_report)


# A confusion matrix shows the number of:
# True Positives (TP): Correctly predicted malignant cases.
# True Negatives (TN): Correctly predicted benign cases.
# False Positives (FP): Benign cases incorrectly predicted as malignant (also known as Type I error).
# False Negatives (FN): Malignant cases incorrectly predicted as benign (also known as Type II error).

# Step 6: Visualization.
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# See
# True Positives (TP): Correctly predicted malignant cases are 40
# True Negatives (TN): Correctly predicted benign cases are 68

# Let's visualize tree now.
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
