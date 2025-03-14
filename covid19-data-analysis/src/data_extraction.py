import pandas as pd
import requests

def download_dataset(url: str, save_path: str):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

def extract_covid19_data(url: str, save_path: str) -> None:
    """
    Extracts COVID-19 data from the specified URL and saves it as a CSV file.

    Parameters:
    url (str): The URL of the COVID-19 dataset.
    save_path (str): The path where the CSV file will be saved.
    """
    try:
        print(f"Attempting to load data from: {url}")
        covid_data = pd.read_csv(url)  # Attempt to load data from the URL
        print(f"Data loaded successfully. Shape: {covid_data.shape}")
        print(covid_data.head())  # Print the first few rows of the data for verification
        
        # Save the dataset to the specified path
        covid_data.to_csv(save_path, index=False)
        print(f"Data extracted and saved to {save_path}")
        
    except pd.errors.ParserError as e:
        print(f"Data parsing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Log the error for further analysis

# Example usage:
if __name__ == "__main__":
    extract_covid19_data(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'D:\\Projects\\Data Analyst Project\\covid19-data-analysis\\data\\covid19_data.csv'
    )
