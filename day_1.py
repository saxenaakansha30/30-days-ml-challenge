# Predict house prices using Simple Linear Regression
# Data is from Sklearn library.

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# Step 1: load the dataset as Pandas Dataframe.
california_housing = fetch_california_housing(as_frame=True)
california_housing_df = california_housing.frame

# Step 2: Select MedInc as data and MedHouseVal as target
X = california_housing_df[['MedInc']]
y = california_housing_df['MedHouseVal']

# Step 3: Split the data, 80% for training and 20% for validating.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model.
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluation.
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse) #  The lower the RMSE, the better your model has performed in predicting the target variable.
print("\nThe Root Mean Squared error is: ", rmse)

# Step 6: Visualization.
plt.scatter(X_val, y_val, label='True Value', alpha=0.5)
plt.plot(X_val, predictions, label='Predicted Values', color='red')
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.legend()
plt.show()