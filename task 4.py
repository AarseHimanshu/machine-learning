# Import libraries
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load dataset (MovieLens sample dataset)
ratings_dict = {
    "userID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "itemID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "rating": [5, 3, 4, 4, 5, 2, 1, 3, 4, 5],
}
df = pd.DataFrame(ratings_dict)

# Define reader format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# Split dataset
trainset, testset = train_test_split(data, test_size=0.2)

# Train Model
model = SVD()
model.fit(trainset)

# Test Model
predictions = model.test(testset)

# Evaluate Model
print("RMSE:", rmse(predictions))
