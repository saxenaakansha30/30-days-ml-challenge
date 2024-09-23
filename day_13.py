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
for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    count = 1
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
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

        count = count + 1
        if count > 10:
            break

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
