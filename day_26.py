# Problem: Time Series Forecasting of electricity consumption using LSTM (Deep Learning Intro)
# Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Step 1: Load the data
data = pd.read_csv('dataset/AEP_hourly.csv')
# print(data.info())
# print(data.head())

# Convert the Datetime column to datetime object.
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Set the Datetime as index.
data.set_index('Datetime', inplace=True)

# Check for missing values.
print(data.isnull().sum())

# Step 2: Handle Missing Values (if any)

# If there would have been any missing values.
# First use iterpolation to predict missing values from neighbour data points.
data = data.interpolate()
# Drop the missing if left any.
data = data.dropna()

# Step 3: Normalize the Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['AEP_MW']])
# Check first five rows of scaled data.
# print(data_scaled[:5])

# Step 4: Create Sequences for LSTM Model

# data: The entire time series data (in this case, the scaled electricity consumption data).
# time_steps: The number of previous time steps (hours) you want to use as input to predict the next time step.
def create_sequences(data, time_steps):
    sequences = [] # This list will store the input sequences (i.e., the previous 60 hours of electricity consumption).
    target = [] # This list will store the corresponding target values.

    for i in range(len(data) - time_steps):
        sequences.append(data[i: i + time_steps])
        target.append(data[i + time_steps])

    return np.array(sequences), np.array(target)

time_steps = 60
# Create sequences
X, y = create_sequences(data_scaled, time_steps)

# Check the shape of sequences
# print(X.shape, y.shape)
# Output: (121213, 60, 1) (121213, 1)

# Step 5: Split the data into training and validation datasets
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Check the shape of train and validation dataset.
# print(X_train.shape, X_val.shape)
# print(y_train.shape, y_val.shape)

# Step 6: Build and Train the LSTM Model
model = Sequential()

# Add an LSTM layer with 50 units
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Add a Dense layer to output a single value.
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Step 7: Evaluate the Model
# Make prediction
predictions = model.predict(X_val)

# Transforms the predicted values from the scaled range (0-1) back to the original range, which represents real
# electricity consumption values in MW.
predictions = scaler.inverse_transform(predictions)
y_val_org = scaler.inverse_transform(y_val.reshape(-1, 1))

# Visualization
plt.figure(figsize=(10, 7))
plt.plot(y_val_org, label="Actual Consumption")
plt.plot(predictions, label="Predicted Consumption")
plt.xlabel('Time')
plt.ylabel('Consumption (MW)')
plt.title('Electricity Consumption Forecasting')
plt.legend()
plt.show()