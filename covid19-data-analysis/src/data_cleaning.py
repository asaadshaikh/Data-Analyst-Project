import pandas as pd

def load_data(file_path):
    """Load the COVID-19 dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame by filling them with the column mean.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled.
    """
    return df.fillna(df.mean())

def standardize_date_format(data):
    """Standardize the date format in the dataset."""
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    return data

def clean_data(file_path):
    """Load, clean, and return the COVID-19 dataset."""
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = standardize_date_format(data)
    return data
