from data_load import train_df

def preprocess_train_data(train_df):
    '''
    Preprocess the training data by dropping unnecessary features and removing outliers.
    '''
    # Save the 'Id' column for both train and test dataframes
    train_ID = train_df['Id']

    # Drop the 'Id' column from both train and test dataframes since it's not necessary for the prediction process.
    train_df.drop("Id", axis=1, inplace=True)

    # Drop the rows with GrLivArea greater than 4000 and SalePrice less than 300000
    train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)

    return train_df

