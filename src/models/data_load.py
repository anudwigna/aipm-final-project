import os
import pandas as pd


def load_data():
    project_path = os.path.dirname(os.path.abspath("requirements.txt"))
    train_df = pd.read_csv(f"{project_path}/src/data/train.csv")
    test_df = pd.read_csv(f"{project_path}/src/data/test.csv")
    return train_df, test_df

train_df, test_df = load_data()
