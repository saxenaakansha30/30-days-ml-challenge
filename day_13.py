# Problem is: Build a music genre classifier using audio features extraction
# Dataset from: https://github.com/ez2rok/music-genre-classification/blob/main/how-to-download-dataset.md

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data and extract features.

# Define the path to dataset.
dataset_path = 'dataset/genres/'

# Initialize lists to store features and labels.
features = []
labels = []

# Loop through each genre folder and load audio files.
# ***Corrupted File Issue: During the process, we encountered an issue with the file /jazz/jazz.00054.wav,
# which was corrupted and had to be removed from the dataset before training.
for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        print(file_path)
        # Load the audio file and extract features
        y, sr = librosa.load(file_path, duration=30) # Load a 30 sec audio clip
        # Sample output:
        # y: [ 0.03451538  0.04815674  0.06430054 ... -0.03909302 -0.02001953 0.05392456]
        # sr: 22050

        # Extract features.
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc=13).mean(axis=1)
        chroma = librosa.feature.chroma_stft(y = y, sr = sr).mean(axis=1)
        contrast = librosa.feature.spectral_contrast(y = y, sr = sr).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y).mean()

        # Append to features labels list
        features.append(np.hstack([mfcc, chroma, contrast, zcr]))
        labels.append(genre)

# Convert features and labels to dataframe
X = pd.DataFrame(features)
y = pd.Series(labels)

# Step 2:  Split the data: training and testing datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build and Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy Score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions, zero_division=1)
print("Classification report:\n", classification_report)

# Step 5: Visualization
plt.figure(figsize=(10, 7
                    ))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Music Genre Classification')
plt.show()

# Model Performance:

# Accuracy Score:
#  0.635
# Confusion Matrix:
#  [[17  0  1  1  0  1  0  0  1  1]
#  [ 0 26  1  0  1  0  0  0  0  0]
#  [ 2  0 13  2  0  0  0  1  3  1]
#  [ 2  0  1 12  3  1  0  0  0  5]
#  [ 0  0  0  2 11  0  1  1  2  3]
#  [ 2  3  0  1  0 10  0  0  3  0]
#  [ 1  0  0  0  0  0 11  0  0  0]
#  [ 0  0  2  1  1  2  0 14  1  0]
#  [ 1  0  2  0  4  0  0  1  7  0]
#  [ 4  0  2  5  0  0  0  0  0  6]]
# Classification report:
#                precision    recall  f1-score   support
#
#        blues       0.59      0.77      0.67        22
#    classical       0.90      0.93      0.91        28
#      country       0.59      0.59      0.59        22
#        disco       0.50      0.50      0.50        24
#       hiphop       0.55      0.55      0.55        20
#         jazz       0.71      0.53      0.61        19
#        metal       0.92      0.92      0.92        12
#          pop       0.82      0.67      0.74        21
#       reggae       0.41      0.47      0.44        15
#         rock       0.38      0.35      0.36        17
#
#     accuracy                           0.64       200
#    macro avg       0.64      0.63      0.63       200
# weighted avg       0.64      0.64      0.63       200
