import pandas as pd
import os

def load_data():
    project_path = os.path.dirname(os.path.abspath("requirements.txt"))
    #Import the training and test data from CSV files into pandas dataframes
    train_df = pd.read_csv(f"{project_path}/src/data/train.csv")
    return train_df

train_df= load_data()
