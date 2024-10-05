# Problem: Sentiment Analysis of customer reviews using traditional NLP techniques
# Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data
data = pd.read_csv('dataset/IMDB_Dataset.csv')
# print(data.info())
# print(data.head())

# Step 2: Data Preprocessing
# Function to clean the text data.
def preprocesss_text(text):
    # Remove the HTMl tags and special characters.
    text = re.sub(r'<.*?>', '', text)  # <html> <body> etc
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lower.
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove the stopwords and apply stemming.
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens] # Applied stemming.

    return ' '.join(tokens)

# Apply preprocessing on the reviews feature
data['cleaned_review'] = data['review'].apply(preprocesss_text)

# Step 3: Split the data into features and target datasets.
X = data['cleaned_review'] # Feature
y = data['sentiment']

# Step 4: Split the dataset into training(80%) and validation(20%) dataset.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Step 6: Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Make Prediction and Evaluate.
predictions = model.predict(X_val_tfidf)

accuracy_score = accuracy_score(y_val, predictions)
confusion_matrix = confusion_matrix(y_val, predictions)
classification_report = classification_report(y_val, predictions)

print(f"Accuracy Score: \n {accuracy_score}")
print(f"Confusion Matrix: \n {confusion_matrix}")
print(f"Classification Report: \n {classification_report}")

# Step 8: Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()

# Output
# Accuracy Score:
#  0.8848
# Confusion Matrix:
#  [[4304  657]
#  [ 495 4544]]
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.90      0.87      0.88      4961
#     positive       0.87      0.90      0.89      5039
#
#     accuracy                           0.88     10000
#    macro avg       0.89      0.88      0.88     10000
# weighted avg       0.89      0.88      0.88     10000
