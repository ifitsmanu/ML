
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def convert_to_float(value):
    if isinstance(value, str):
        # Extract numbers from the string using regex
        numbers = re.findall(r'\d+(?:\.\d+)?', value)
        if numbers:
            # Return the average of the extracted numbers
            return sum(float(num) for num in numbers) / len(numbers)
        else:
            return 0.0
    return value

def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan


def clean_job_titles(df):
    return df

def preprocess_data(df):
    # Handle non-numeric columns, for example using label encoding
    le_state = LabelEncoder()
    le_zip_code = LabelEncoder()

    # Replace 'state' with the encoded values
    df['state'] = le_state.fit_transform(df['state'])

    # Replace 'zip_code' with the encoded values
    df['zip_code'] = le_zip_code.fit_transform(df['zip_code'])

    return df


def load_and_process_data(filenames):
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        print(f"Shape of the DataFrame loaded from {filename}:", df.shape)
        df = clean_job_titles(df)
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    print("Shape of the DataFrame after concatenating:", combined_df.shape)
    combined_df = combined_df[[ 'job_title', 'job_description', 'min_pay', 'max_pay', 'commission', 'state', 'zip_code', 'count_apps_id']]
    print("Shape of the DataFrame after selecting columns:", combined_df.shape)
    
    # Converting non-string values to string in 'job_title' and 'job_description'
    for col in ['job_title', 'job_description']:
        combined_df[col] = combined_df[col].astype(str)

    assert combined_df['state'].dtype.name == 'category' or combined_df['state'].dtype.name == 'object', "state is not a categorical variable"
    # Fill NaN values with appropriate default values
    combined_df['count_apps_id'] = combined_df['count_apps_id'].replace('missing', np.nan).astype(float).astype('Int64')
    combined_df.dropna(subset=['count_apps_id'], inplace=True)  # Replace missing values with 0
    combined_df['min_pay'] = combined_df['min_pay'].apply(convert_to_float)
    combined_df['max_pay'] = combined_df['max_pay'].apply(convert_to_float)
    combined_df['commission'] = combined_df['commission'].apply(convert_to_float)
    combined_df['state'].fillna('Unknown', inplace=True)
    combined_df['zip_code'].fillna(0, inplace=True)
    combined_df['job_description'].fillna('', inplace=True)

    print("Shape of the DataFrame after filling NaN values:", combined_df.shape)
    return combined_df
