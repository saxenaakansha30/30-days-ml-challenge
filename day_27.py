# Problem: Image Classification with a small CNN on CIFAR-10 dataset
# Dataset: from tensorflow.keras.datasets import cifar10

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np


# Step 1: Load the data
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# Normalize the image data.
# Scale the data to [0, 1] range.
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Apply one-hot encoding to labels.
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# Check the shape of the data.
print(X_train.shape, X_val.shape)
print(y_train.shape, y_val.shape)

# Output:
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# (50000, 10) (10000, 10)

# Step 2: Build a small CNN Architecture
model = Sequential()
# Add first convolutional layer.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2)) # To reduce the number of parameters.

# Add second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Add third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten Layer: This layer flattens the 3D output from the convolutional layers into a 1D vector
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Dropout is a regularization technique used to prevent overfitting
model.add(Dropout(0.5))

# Output Layer:
model.add(Dense(10, activation='softmax'))

model.summary()

# Step 3: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history  = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Step 5: Evaluate the model and Visualize the results
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validate Accuracy: {val_acc}")

# Visualize
plt.figure(figsize=(10, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Data Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Step 6: Make Prediction
# Pick a random image from the test set
random_idx = np.random.randint(0, len(X_val))
random_image = X_val[random_idx]
# Make prediction
prediction = model.predict(np.expand_dims(random_image, axis=0))
prediction_class = np.argmax(prediction, axis=1)
print(f"Prediction array: {prediction}")
print(f"Actual Class of the random image: {np.argmax(y_val[random_idx])}")
print(f"Prediction class: {prediction_class[0]}")

# Visualize the image with the prediction
plt.imshow(random_image)
plt.title(f"Predicted: {prediction_class[0]}")
plt.show()