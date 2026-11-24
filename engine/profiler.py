import pandas as pd 
def generate_profile(df):

    profile={}

    profile["total_rows"]=df.shape[0]
    profile["total_columns"]=df.shape[1]

    profile["missing_values"]=df.isnull().sum()
    profile["total_missing_values"]=int(df.isnull().sum().sum())

    profile["duplicate_rows"]=int(df.duplicated().sum())

    profile["data_types"]=df.dtypes.astype(str)
    profile["unique_values"]=df.nunique()
    profile["numeric_summary"]=df.describe()

    return profile

     