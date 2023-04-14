import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from feature_engineering import train_df, test_df,y_train, ntrain, test_ID

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)



def scale_continuous_features(train_df, test_df):

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

    return train_df, test_df

train_df, test_df = scale_continuous_features(train_df, test_df)
print(train_df.head())
print(test_df.head())



def split_train_val(all_data, y_train, ntrain):
    # Your dataset with input features
    X = all_data.iloc[:ntrain, :]

    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)
    
    return X, X_train, X_val, y_train, y_val


X, X_train, X_val, y_train, y_val = split_train_val(train_df, y_train, ntrain)




def gradient_boosting_regressor(X_train, y_train, X_val, y_val):
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_val_preds = gb.predict(X_val)
    gb_rmsle = compute_rmsle(y_val, gb_val_preds)
    return gb_rmsle

gb_rmsle=gradient_boosting_regressor(X_train, y_train, X_val, y_val)

print(gb_rmsle)



# Train the model
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# Make predictions on the test set
test_preds = gb.predict(test_df)

# Save predictions to a file
submission_df = pd.DataFrame({'Id': test_ID, 'SalePrice': test_preds})
submission_df.to_csv('submission.csv', index=False)

# Print some of the predictions
print(submission_df.head(10))
