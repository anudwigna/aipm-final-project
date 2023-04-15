import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.impute import SimpleImputer

from data_processing import train_df, train_ID,test_df, test_ID

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

def extract_target_variable(train_df, test_df):

    ntrain = train_df.shape[0]
    ntest = test_df.shape[0]

    y_train = train_df.SalePrice.values

    all_data = pd.concat([train_df, test_df], axis=0)
    all_data = all_data.reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    train_df = all_data.iloc[:ntrain, :]
    return all_data, train_df, y_train

all_data, train_df, y_train = extract_target_variable(train_df, test_df)



def calculate_missing_data(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)
    all_data_na = all_data_na.sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Column': all_data_na.index, 'Missing Ratio (%)': all_data_na.values})
    return missing_data

missing_data = calculate_missing_data(all_data)


def impute_missing_data(all_data):
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
    return all_data
all_data = impute_missing_data(all_data)


def add_total_sf_feature(all_data):
    
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

    return all_data

all_data = add_total_sf_feature(all_data)


def one_hot_encode(all_data):
    """
    One-hot encode categorical features using pandas get_dummies() function
    """
    all_data = pd.get_dummies(all_data)
    train_df = all_data.iloc[:ntrain, :]
    test_df = all_data.iloc[ntrain:, :]
    return all_data, train_df, test_df

all_data, train_df, test_df = one_hot_encode(all_data)

