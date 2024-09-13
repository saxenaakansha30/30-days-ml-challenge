# Day 3, Problem: Recognize handwritten digits with k-Nearest Neighbors on MNIST
# This is a classification problem.

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the MNIST data
mnist_data = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')

# The image of size 28x28 pixel is already flattened into 784 feature vector.
# Step 2: Preprocess the Data
X = mnist_data.data
y = mnist_data.target

# Step 3: Normalize the data. I already see the data in 0,1 format but
# ChatGPT says it is better to normalize the data again as KNeighborsClassifier Algo works on that.
X /= 255.0

# Step 4: Split into training(80%) and validation(20%) set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the model
# We start with K=3, which is normal and initial choice. Later experiment with it.
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Make Predictions and Evaluate the Model.
predictions = model.predict(X_val)
# Evaluate actual value against predicted values.
accuracy_score = accuracy_score(y_val, predictions)
print("\nAccuracy:\n", accuracy_score)
# You can see True Positives in diagonally and all other FP, FN, TN in other columns.
# The TP are in good number that is why the accuracy is 0.97
confusion_matrix = confusion_matrix(y_val, predictions)
print("\nConfusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions)
print("\nClassfication Report:\n", classification_report)

# Step 7: Visualization.
plt.figure(figsize=(10,7)) # I will go with Day 2 plot size.
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Greens', xticklabels=mnist_data.target_names, yticklabels=mnist_data.target_names)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()

# Let's change the K to 1 and see what happens.
# Accuracy improved, Changes in the confusion matrix are difficult to analyze at this moment. Too many numbers.

# Let's increase the K to 10.
# Accuracy dropped significantly to: 0.9657857142857142

# At K=5, Accuracy: 0.9700714285714286

# AT K=3 the accuracy is best:  0.9712857142857143

# Let's see at K=2, Accuracy:  0.9642142857142857

# so K=3 is best for our problem.