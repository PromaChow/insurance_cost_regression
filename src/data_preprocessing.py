import pandas as pd
import numpy as np
from datetime import datetime

def pre_processing_data(df1, df2):
    
    df = pd.merge(df1, df2, on='Customer ID', how='inner')
    df = df.drop(columns=["Customer ID"])
    
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    df = df.drop(columns=['year', 'month', 'date'])

    df = df[~df.eq('?').any(axis=1)]
    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)  

    def clean_tier(entry):
        return int(entry[-1])


    df['Hospital_tier'] = df['Hospital_tier'].apply(clean_tier)
    df['City_tier'] = df['City_tier'].apply(clean_tier)

    df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].replace('No major surgery', 0)
    df['NumberOfMajorSurgeries'] = pd.to_numeric(df['NumberOfMajorSurgeries'], errors='coerce')
    

    def one_hot(df, drop_first=True):
    
        obj_col = df.select_dtypes(include=['object']).columns.tolist()

        for col in obj_col:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=drop_first)], axis=1)
            df = df.drop(col, axis=1)

        return df
    
    df = one_hot(df)
    
    return df

