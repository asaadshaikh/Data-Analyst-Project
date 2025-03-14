import pandas as pd

def load_data(file_path):
    """Load COVID-19 data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def calculate_statistics(data):
    """Calculate basic statistics for key metrics and return additional metrics."""
    stats = {
        'mean_cases': data['daily_cases'].mean(),
        'median_cases': data['daily_cases'].median(),
        'std_cases': data['daily_cases'].std(),
        'mean_deaths': data['daily_deaths'].mean(),
        'median_deaths': data['daily_deaths'].median(),
        'std_deaths': data['daily_deaths'].std(),
        'mean_recoveries': data['daily_recoveries'].mean(),
        'median_recoveries': data['daily_recoveries'].median(),
        'std_recoveries': data['daily_recoveries'].std(),
        'total_cases': data['daily_cases'].sum(),  # New metric
        'total_deaths': data['daily_deaths'].sum(),  # New metric
    }
    return stats

def identify_trends(data):
    """Identify trends and patterns in the data."""
    data['date'] = pd.to_datetime(data['date'])
    trends = data.groupby('date').agg({
        'daily_cases': 'sum',
        'daily_deaths': 'sum',
        'daily_recoveries': 'sum'
    }).reset_index()
    return trends

def group_by_region(data):
    """Group data by regions or countries for comparative analysis."""
    grouped_data = data.groupby('region').agg({
        'total_cases': 'sum',
        'total_deaths': 'sum',
        'total_recoveries': 'sum'
    }).reset_index()
    return grouped_data

def perform_eda(df: pd.DataFrame):
    """
    Perform basic exploratory data analysis on the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame for EDA.
    """
    print("Basic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nData Types:")
    print(df.dtypes)

def main():
    """Main function to execute exploratory data analysis."""
    file_path = 'D:\\Projects\\Data Analyst Project\\covid19-data-analysis\\data\\covid19_data.csv'
    data = load_data(file_path)
    
    stats = calculate_statistics(data)
    trends = identify_trends(data)
    grouped_data = group_by_region(data)
    
    print("Statistics:", stats)
    print("Trends:", trends)
    print("Grouped Data:", grouped_data)

if __name__ == "__main__":
    main()
