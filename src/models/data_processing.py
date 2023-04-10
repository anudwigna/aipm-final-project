from data_load import train_df

'''
Dropping the 'Id' Feature
'''
#Save the 'Id' column for both train and test dataframes
train_ID = train_df['Id']

#Drop the 'Id' column from both train and test dataframes since it's not necessary for the prediction process.
train_df.drop("Id", axis = 1, inplace = True)

'''
Remove Outliers: Drop Rows with Large GrLivArea and Low SalePrice
'''
# Drop the rows with GrLivArea greater than 4000 and SalePrice less than 300000
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)



