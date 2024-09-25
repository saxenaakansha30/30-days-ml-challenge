# Problem: Predict house prices with XGBoost
# Dataset: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

# Step 1: Load the data
data = pd.read_csv('dataset/AmesHousing.csv')
# print(data)

# Step 2: Preprocess the data
# Handle missing values:
# Fill missing numerical values with the columns median
# Fill categorical vales with 'None'
# Encode the categorical data using One Hot Encoding

for col in data.columns:
    if data[col].dtypes in [np.int64, np.float64]:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna('None', inplace=True)

# One Hot Encoding used to encode the categorical data.
data_encoded = pd.get_dummies(data, drop_first=True)

# Define the target and feature sets.
X = data_encoded.drop('SalePrice', axis=1)
y = data_encoded['SalePrice']

# Step 3: Split the data.
# Split the data into 80% for training and 20% for validation.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the model.
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make prediction and Evaluate
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
print("Root mean square error: ", rmse)

# Step 6: Visualization
# XGBoost provides a built-in method to plot feature importance, which can give insights into the most
# important features for predicting house prices.
plt.figure(figsize=(7, 5))
xgboost.plot_importance(model, max_num_features=10)
plt.title('Top 10 Important Features')
plt.show()
