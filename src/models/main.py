import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from data_load import load_data
from data_processing import preprocess_train_data
from feature_engineering import extract_target_variable,calculate_missing_data,impute_missing_data,add_total_sf_feature,one_hot_encode
from model import compute_rmsle,scale_continuous_features,split_train_val,gradient_boosting_regressor,train_and_predict
# Load the data
train_df, test_df = load_data()

# Preprocess the train data
train_df, train_ID, test_df, test_ID = preprocess_train_data(train_df, test_df)

# Extract target variable and concatenate train and test data
all_data, train_df, y_train = extract_target_variable(train_df, test_df)

# Calculate missing data
missing_data = calculate_missing_data(all_data)

# Impute missing data
all_data = impute_missing_data(all_data)

# Add TotalSF feature
all_data = add_total_sf_feature(all_data)

# One-hot encode categorical features
ntrain = train_df.shape[0]
all_data, train_df, test_df = one_hot_encode(all_data)

# Scale continuous features
train_df, test_df = scale_continuous_features(train_df, test_df)

# Split train set into train and validation sets
X, X_train, X_val, y_train, y_val = split_train_val(train_df, y_train, ntrain)

# Calculate Gradient Boosting Regressor RMSLE
gb_rmsle = gradient_boosting_regressor(X_train, y_train, X_val, y_val)

# Call the train_and_predict function
train_and_predict(X_train, y_train, test_df, test_ID)

