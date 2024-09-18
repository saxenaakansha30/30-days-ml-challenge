# Problem: Detecting fake news with a PassiveAggressive Classifier and TfidfVectorizer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data.
# Download the datsets from Kaggle 'Fake News Detection Datasets' and put in dataset directory.
fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

# Step 2: Prepare the data
fake_df['label'] = 0 # Label 0 is for Fake news
true_df['label'] = 1 # Label 1 is for True news
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# Step 3: Preprocess the data: Create feature and target datasets.
X = df['text'] # Feature
y = df['label'] # Target

# Step 4: Convert the text data into numerical form using TfidfVectorizer.
# TfidfVectorizer depicts the importance and uniqueness of the word in the dataset.
tf_idf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7) # Ignore if the words appears in 70% or more of the documents.
X_tf_idf = tf_idf_vectorizer.fit_transform(X)

# Step 5: Split the Data.
# Split into 80-20 ratio: Training (80%) and Validation(20%) datasets.
X_train, X_val, y_train, y_val = train_test_split(X_tf_idf, y, test_size=0.2, random_state=42)

# Step 6: Create and train the model
# Each iteration allows the model to learn from mistakes it made in the previous iteration.
# The PassiveAggressiveClassifier gets updated "aggressively" when it misclassifies an example,
# adjusting its decision boundary to fix the mistake.
model = PassiveAggressiveClassifier(max_iter=10)
model.fit(X_train, y_train)

# Step 7: Make prediction and Evaluate
predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions)
print("Classfication Report:\n", classification_report)


# Step 8: Visualization
plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()