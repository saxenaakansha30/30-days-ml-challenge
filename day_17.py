# Problem: Predict diabetes onset using Decision Trees and Random Forests
# Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data
data = pd.read_csv('dataset/diabetes.csv')
#print(data.head())

# Step 2: Split the data into features and target
X = data.drop('Outcome', axis=1) # Feature
y = data['Outcome'] # Target


# Step 3: Split into training and validation dataset.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Step 5: Build and Train the Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=5, random_state=42)
random_forest.fit(X_train, y_train)

# Step 6: Make the predictions.
dt_predictions = decision_tree.predict(X_val)
rf_predictions = random_forest.predict(X_val)

# Step 7: Evaluate the models.

# Decision Tree.
dt_accuracy = accuracy_score(y_val, dt_predictions)
dt_confusion_matrix = confusion_matrix(y_val, dt_predictions)
dt_classification_report = classification_report(y_val, dt_predictions)

print(f"Decision Tree Accuracy Score: {dt_accuracy}")
print(f"Decision Tree Confusion Matrix: {dt_confusion_matrix}")
print(f"Decision Tree Classification Report: {dt_classification_report}")

# Random Forest
rf_accuracy = accuracy_score(y_val, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_val, rf_predictions)
rf_classification_report = classification_report(y_val, rf_predictions)

print(f"Random Forest Accuracy Score: {rf_accuracy}")
print(f"Random Forest Confusion Matrix: {rf_confusion_matrix}")
print(f"Random Forest Classification Report: {rf_classification_report}")

# Step 7: Visualization
# Decision Tree
plt.figure(figsize=(7,5))
sns.heatmap(dt_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Random Forest
plt.figure(figsize=(7,5))
sns.heatmap(rf_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Random Forest Confusion Matrix')
plt.show()

# The Random Forest Model is not performing good in comparison to Decision Tree.
# Let's explore which parameters are good for training it, using Hyperparameter Tuning.
#
# # Define the parameter grid.
# param_grid = {
#     'n_estimators': [100, 200, 300],  # The number of trees in the forest.
#     'max_depth': [None, 10, 20, 30],  # The maximum depth of a tree.
#     'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node.
#     'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node.
# }
#
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# # Fit the model.
# grid_search.fit(X_train, y_train)
# # Get the best parameters and accuracy.
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Score: {grid_search.best_score_}")
#
# Output:
# Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# Best Score: 0.7834199653471945