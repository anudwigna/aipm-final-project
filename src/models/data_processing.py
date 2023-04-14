from data_load import train_df,test_df

def preprocess_train_data(train_df,test_df):
    train_ID = train_df['Id']
    test_ID = test_df['Id']
    train_df.drop("Id", axis=1, inplace=True)
    test_df.drop("Id", axis=1, inplace=True)
    train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
    return train_df, train_ID,test_df, test_ID

train_df, train_ID,test_df, test_ID =preprocess_train_data(train_df,test_df)
  
