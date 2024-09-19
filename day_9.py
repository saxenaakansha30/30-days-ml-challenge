# Problem: Forecasting weather with Simple Linear Regression on time series data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Step 1: Load the data
# Download the dataset from Kaggle Daily Temperature Of Major Cities. And put in the dataset directory.
# https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
data = pd.read_csv('dataset/city_temperature.csv', low_memory=False)

# Step 2: Filter the data for the country 'India
india_data = data[(data['Country'] == 'India') & (data['AvgTemperature'] > -99)].copy()

# Step 3: Combine the Month, Day and Year into a single Date column using .loc
india_data.loc[:, 'Date'] = pd.to_datetime(india_data[['Year', 'Month', 'Day']])

# Step 4: Select the relevant columns (Datetime, AvgTem)
rel_india_data = india_data[['Date', 'AvgTemperature']]

# Step 5: Preprocess the data.
# Models like Linear Regression cannot work with missing data, so we need to handle them.
# Weather data is measured daily and missing values might be a very few, So removing the missing values rows will be good here.
rel_india_data = rel_india_data.dropna()

# Date_ordinal column will contain integer values, which represent the date as a number, making it usable for the Linear Regression model.
# pd.Timestamp('1995-01-01').toordinal() might return 729216.
rel_india_data['Date_ordinal'] = rel_india_data['Date'].map(pd.Timestamp.toordinal)

X = rel_india_data[['Date_ordinal']] # Feature
y = rel_india_data['AvgTemperature'] # Target

# Step 6: Split the data
# Split into 80-20 ratio, training (80%) and validation (20%) datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 7: Build and Train the model.
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions and Evaluate
predictions = model.predict(X_val)
mean_squared_error = mean_squared_error(y_val, predictions)
print(mean_squared_error)

# Step 9: Visualization
plt.figure(figsize=(7, 5))
plt.scatter(X_val, y_val, color='blue', label='Actual Temperature')
plt.plot(X_val, predictions, color='red', linewidth=2, label='Predicted Temperature')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Temperature')
plt.title('Weather Forecast for India: Actual vs Predicted')
plt.legend()
plt.show()

# The accuracy is coming as 74%
# The model performs poorly.
# What Can We Do?

# Introduce Complexity: If you want to capture more accurate weather forecasts, we might need a more sophisticated model,
# such as Polynomial Regression (to capture nonlinear trends) or Time Series Models (like ARIMA or Prophet)
# that can account for seasonal patterns.

# Add Features: Simple Linear Regression is based only on the date. Adding more features (e.g., previous day’s temperature,
# humidity, pressure, etc.) can improve predictions.