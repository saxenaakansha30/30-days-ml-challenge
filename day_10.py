import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
ratings = pd.read_csv('dataset/ratings.csv')
movies = pd.read_csv('dataset/movies.csv')

# View the structure of the dataset
# print(ratings.head())
# print(movies.head())

# Step 2: Create user item matrix
# We’ll create a matrix where rows represent users, columns represent movies,
# and the values are the ratings given by the users to the movies.
# Missing values indicate that the user hasn’t rated the movie yet.
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill the missing value with 0.
user_item_matrix.fillna(0, inplace=True)
#print(user_item_matrix)

# Step 3: Implement Collaborative filtering.
# We’ll use Cosine Similarity to find similar users or items.
# For user-based collaborative filtering, we calculate the similarity between users,
# and for item-based filtering, we calculate the similarity between movies.
# Let’s use item-based collaborative filtering (we recommend movies similar to those the user already liked):
# Example: Recommend a comedy movie because user has already liked a comedy movie.
# We will use cosine similarity becuase it focusses on pattern similarity rather than absolute ratings.

# Calculate cosine similarity between movies.
item_simlarity = cosine_similarity(user_item_matrix.T)
# This matrix shows which movies were rated by which users, and the ratings given.
# We did transpose because:When calculating item-based similarity, we want to compare movies (items) with each other, not users.
item_simlarity_df = pd.DataFrame(item_simlarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Step 4: Make predictions based on similarity
# We are using item_similarity calculated above to predict the ratings of the movies the users have not rated yet.
# We do so by using cosine_similarity, that is:
# If two movies are similar, the rating for one can help predict the rating for the other.
# For example, if a user rated Movie A and Movie B is very similar to Movie A, we can predict how the user might rate Movie B based on their rating of Movie A.
def predict_ratings(user_item_matrix, item_similarity_matrix):
    return np.dot(user_item_matrix, item_similarity_matrix) / np.abs(item_similarity_matrix).sum(axis=1)

# Make prediction
predicted_ratings = predict_ratings(user_item_matrix.values, item_simlarity) # which pass the user ratings for each movie
# Convert it to Dataframe for better readability:
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
# print(predicted_ratings_df)


# Step 4: Evaluate the model
# Flatten the matrix and calculate root mean square error.
true_ratings = user_item_matrix.values.flatten()
predicted_ratings = predicted_ratings_df.values.flatten()

# Calculate root mean square error:
rmse = mean_squared_error(true_ratings[true_ratings > 0], predicted_ratings[true_ratings > 0])
print("The root mean square error: ", rmse)