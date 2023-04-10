import pandas as pd
import os

def load_data():
    project_path = os.path.dirname(os.path.abspath("requirements.txt"))
    #Import the training and test data from CSV files into pandas dataframes
    train_df = pd.read_csv(f"{project_path}/src/data/train.csv")
    test_df = pd.read_csv(f"{project_path}/src/data/test.csv")
    return train_df, test_df

train_df, test_df = load_data()

# #Print the shape of the training and test dataframes to verify the number of rows and columns
# print(f"Training data shape: {train_df.shape}")
# print(f"Test data shape: {test_df.shape}")

# #View the first few rows of the training data to check the data has been loaded properly
# print(train_df.head(5))

# #View the first few rows of the test data to check the data has been loaded properly
# print(test_df.head(5))