import os
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib
import mlflow
import mlflow.sklearn

project_path = os.path.dirname(os.path.abspath("requirements.txt"))
train_df = pd.read_csv(f"{project_path}/src/data/train.csv")
test_df = pd.read_csv(f"{project_path}/src/data/test.csv")

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

y_train = train_df.SalePrice.values

all_data = pd.concat([train_df, test_df], axis=0)
all_data = all_data.reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
train_df = all_data.iloc[:ntrain, :]

# Feature engineering
categorical_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
                        "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                        "BsmtFinType2", "MasVnrType"]

for col in categorical_cols:
    all_data[col] = all_data[col].fillna("None")

# Impute missing values with the median for LotFrontage feature
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Impute missing values with 0 for numerical features
numerical_cols = ["MasVnrArea", "GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
                "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
for col in numerical_cols:
    all_data[col] = all_data[col].fillna(0)

# Impute missing values with the mode for remaining features
imputer = SimpleImputer(strategy='most_frequent')
remaining_cols = all_data.columns[all_data.isna().any()].tolist()
for col in remaining_cols:
    all_data[col] = imputer.fit_transform(all_data[[col]]).ravel()


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']    

# Get the indices of all numeric features
numeric_feats = all_data.select_dtypes(include=['int64', 'float64']).columns

# Calculate the skewness of each numeric feature and sort them in descending order
skewness = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# Calculate the skewness of each numeric feature and sort them in descending order
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

# Select only the features with a skewness greater than 0.75
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

    # Define the transformation parameter lambda
lam = 0.15

# Apply the Box Cox transformation to each skewed feature
for feat in skewed_feats.index:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)

train_df = all_data.iloc[:ntrain, :]
test_df = all_data.iloc[ntrain:, :]

# Create independent copies of train_df and test_df
train_df = train_df.copy()
test_df = test_df.copy()

# Get the names of the continuous features
cont_features = ['GrLivArea', 'TotalSF']

# Scale the continuous features
scaler = StandardScaler()

# Use .loc[] to update the training DataFrame
train_df.loc[:, cont_features] = scaler.fit_transform(train_df.loc[:, cont_features])

# Use the same scaler to transform the test DataFrame
test_df.loc[:, cont_features] = scaler.transform(test_df.loc[:, cont_features])

X = all_data.iloc[:ntrain, :]

# Split train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)

X_train.to_parquet(f"{project_path}/models/x_train.parquet")

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

def print_something(abc):
    print(abc)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("learning_rate", type=float, default=0.1)
parser.add_argument("n_estimators", type=int, default=100)

args = parser.parse_args()

if args.learning_rate is None:
    args.learning_rate = 0.1

if args.n_estimators is None:
    args.n_estimators = 100

learning_rate = args.learning_rate
n_estimators = args.n_estimators

with mlflow.start_run():
    # n_estimators=100 
    # learning_rate=0.1
    gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    gb.fit(X_train, y_train)

    gb_val_preds = gb.predict(X_val)
    gb_rmsle = compute_rmsle(y_val, gb_val_preds)

    print(f"Gradient Boosting Algorithm => {n_estimators} estimators, {learning_rate} learning rate")
    print(f"RMSLE: {gb_rmsle}")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)

    mlflow.log_metric("rmsle", gb_rmsle)

    joblib.dump(gb, f"{project_path}/models/gb_model.joblib")
    # mlflow.log_artifact("gb_model", artifact_path="models")
