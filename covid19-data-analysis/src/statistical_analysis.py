import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the COVID-19 dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def correlation_analysis(data):
    """Perform correlation analysis on the dataset."""
    correlation_matrix = data.corr()
    return correlation_matrix

def calculate_growth_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the growth rate of COVID-19 cases.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing daily cases.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for growth rate.
    """
    if 'daily_cases' not in data.columns:
        raise ValueError("The DataFrame must contain a 'daily_cases' column.")
    data['growth_rate'] = data['daily_cases'].pct_change() * 100
    return data

def calculate_doubling_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the doubling time of COVID-19 cases.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing daily cases.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for doubling time.
    """
    if 'growth_rate' not in data.columns:
        raise ValueError("The DataFrame must contain a 'growth_rate' column.")
    data['doubling_time'] = np.log(2) / np.log(1 + data['growth_rate'] / 100)
    return data

def perform_statistical_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical analysis on the COVID-19 data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing daily cases.

    Returns:
    pd.DataFrame: The DataFrame with additional columns for growth rate and doubling time.
    """
    data = calculate_growth_rate(data)
    data = calculate_doubling_time(data)
    return data

def main():
    """Main function to execute statistical analysis."""
    file_path = 'data/covid19_data.csv'
    data = load_data(file_path)
    
    # Perform correlation analysis
    correlation = correlation_analysis(data)
    print("Correlation Matrix:\n", correlation)
    
    # Perform statistical analysis
    data = perform_statistical_analysis(data)
    
    # Display the updated data
    print(data[['date', 'daily_cases', 'growth_rate', 'doubling_time']])

if __name__ == "__main__":
    main()
