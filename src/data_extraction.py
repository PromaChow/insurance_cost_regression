import pandas as pd
import os

def extract_data():
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))
    data_directory = os.path.join(parent_directory, 'data')

    data_path1 = os.path.join(data_directory,'hospitalisation_details.csv')
    data_path2 = os.path.join(data_directory,'medical_examinations.csv')

    df1 = pd.read_csv(data_path1)
    df2 = pd.read_csv(data_path2)

    return df1, df2

