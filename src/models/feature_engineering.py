from data_load import train_df
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.special import boxcox1p

def preprocess_train_data(train_df):
    '''
    Preprocess the training data by extracting the target variable and dropping unnecessary features.
    '''
    # Get the number of rows in the train dataset
    ntrain = train_df.shape[0]

    # Extract the target variable from the train dataset
    y_train = train_df.SalePrice.values

    # Copy the train dataset into a new dataframe
    all_data = train_df.copy()

    # Remove the 'SalePrice' column from the concatenated dataframe
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    # Reset the index of the concatenated dataframe
    all_data = all_data.reset_index(drop=True)

    return all_data, y_train

def calculate_missing_data(all_data):
    '''
    Calculate the percentage of missing values in each column of the concatenated dataset
    '''
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

    # Drop the columns with no missing values
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)

    # Sort the columns in descending order of missing values and keep the top 30
    all_data_na = all_data_na.sort_values(ascending=False)[:30]

    # Create a dataframe to store the missing data information
    missing_data = pd.DataFrame({'Column': all_data_na.index, 'Missing Ratio (%)': all_data_na.values})

    return missing_data

def impute_missing_data(all_data):
    '''
    Impute missing values in the concatenated dataset
    '''
    # Impute missing values with 'None' for categorical features
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
        all_data[col] = imputer.fit_transform(all_data[[col]])

    return all_data


def add_total_sf_feature(all_data):
    '''
    Add total square footage feature
    '''
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    return all_data


def transform_skewed_features(all_data):
    '''
    Transform skewed features using Box-Cox transformation
    '''
    # Get the numeric features
    numeric_feats = all_data.select_dtypes(include=['int64', 'float64']).columns

    # Calculate the skewness of each numeric feature and sort them in descending order
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    # Select only the features with a skewness greater than 0.75
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

    # Print the number of skewed features
    print("There are {} skewed numerical features to Box Cox transform".format(skewed_feats.shape[0]))

    # Define the transformation parameter lambda
    lam = 0.15

    # Apply the Box Cox transformation to each skewed feature
    for feat in skewed_feats.index:
        all_data[feat] = boxcox1p(all_data[feat], lam)
    
    return all_data


def one_hot_encode(all_data):
    '''
    One-hot encode categorical features using pandas get_dummies() function
    '''
    all_data = pd.get_dummies(all_data)
    return all_data
