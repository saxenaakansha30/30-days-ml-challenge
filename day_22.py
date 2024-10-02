# Problem: Recommender System with Matrix Factorization
# Dataset: https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise import accuracy


# Step 1: Load the MovieLens dataset.
ratings = pd.read_csv('dataset/ratings.csv')
print(ratings.head())

# Step 2: Prepare the data for Surprise library.
# print(ratings['rating'].min()) # 0.5
# print(ratings['rating'].max()) # 5.0
# Define the rating scale.
reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
# Load the data into the Surprise dataset format
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Step 3: Split data into train-validation datasets
trainset, valset = train_test_split(data, test_size=0.2)

# Step 4: Matrix Factorization using (Singular Value Decomposition) SVD
svd = SVD()
# Perform cross validation to evaluate the model
cross_validate(svd, data, measures=['RMSE', 'MSE'], cv=5, verbose=True)

# Output
#                   Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
# RMSE (testset)    0.8724  0.8690  0.8734  0.8805  0.8733  0.8737  0.0037
# MSE (testset)     0.7611  0.7552  0.7628  0.7752  0.7627  0.7634  0.0065
# Fit time          0.43    0.41    0.40    0.40    0.40    0.41    0.01
# Test time         0.07    0.08    0.08    0.07    0.05    0.07    0.01

# Step 5: Train the model.
trainset_full = data.build_full_trainset()
svd.fit(trainset_full)

# Step 6: Make Predictions
# Unseen for prediction
user_id = 1
movie_id = 6
prediction = svd.predict(user_id, movie_id)
print(f"Prediction for user {user_id} and movie {movie_id} is: {prediction}")
# Output
# Prediction for user 1 and movie 6 is: user: 1          item: 6          r_ui = None   est = 4.49   {'was_impossible': False}
# RMSE: 0.6411

# Step 7: Evaluate
# Test the model on validation data.
val_predictions = svd.test(valset)
# Calculate the root mean square error
rmse = accuracy.rmse(val_predictions)
print(f"Root mean square error for validation data: {rmse}")
# Root mean square error for validation data: 0.6410624356100669
# The difference between the predicted ratings and the actual ratings is around 0.641, which is quite close.