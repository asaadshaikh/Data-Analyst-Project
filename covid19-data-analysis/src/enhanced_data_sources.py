import pandas as pd
import requests
import os
from datetime import datetime

def download_vaccination_data(save_path='../data/vaccination_data.csv'):
    """
    Download COVID-19 vaccination data from Our World in Data.
    
    Parameters:
    save_path (str): Path to save the vaccination data
    
    Returns:
    pd.DataFrame: DataFrame containing vaccination data
    """
    vaccination_url = "https://covid.ourworldindata.org/data/vaccinations/vaccinations.csv"
    
    try:
        print(f"Downloading vaccination data from {vaccination_url}")
        vaccination_data = pd.read_csv(vaccination_url)
        
        # Save data
        vaccination_data.to_csv(save_path, index=False)
        print(f"Vaccination data saved to {save_path}")
        
        return vaccination_data
    
    except Exception as e:
        print(f"Error downloading vaccination data: {e}")
        return None

def download_mobility_data(save_path='../data/mobility_data.csv'):
    """
    Download Google mobility data showing movement trends during the pandemic.
    
    Parameters:
    save_path (str): Path to save the mobility data
    
    Returns:
    pd.DataFrame: DataFrame containing mobility data
    """
    mobility_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    
    try:
        print(f"Downloading mobility data from {mobility_url}")
        # This is a large file, so we'll use chunks to download it
        mobility_data = pd.DataFrame()
        chunks = pd.read_csv(mobility_url, chunksize=10000)
        
        for chunk in chunks:
            mobility_data = pd.concat([mobility_data, chunk])
        
        # Save data
        mobility_data.to_csv(save_path, index=False)
        print(f"Mobility data saved to {save_path}")
        
        return mobility_data
    
    except Exception as e:
        print(f"Error downloading mobility data: {e}")
        return None

def download_policy_data(save_path='../data/policy_data.csv'):
    """
    Download Oxford COVID-19 Government Response Tracker data.
    
    Parameters:
    save_path (str): Path to save the policy data
    
    Returns:
    pd.DataFrame: DataFrame containing policy intervention data
    """
    policy_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
    
    try:
        print(f"Downloading policy data from {policy_url}")
        policy_data = pd.read_csv(policy_url)
        
        # Save data
        policy_data.to_csv(save_path, index=False)
        print(f"Policy data saved to {save_path}")
        
        return policy_data
    
    except Exception as e:
        print(f"Error downloading policy data: {e}")
        return None

def merge_with_covid_data(covid_data_path, vaccination_data_path=None, mobility_data_path=None, policy_data_path=None):
    """
    Merge COVID-19 data with vaccination, mobility, and policy data.
    
    Parameters:
    covid_data_path (str): Path to COVID-19 data
    vaccination_data_path (str): Path to vaccination data
    mobility_data_path (str): Path to mobility data
    policy_data_path (str): Path to policy data
    
    Returns:
    pd.DataFrame: Merged DataFrame
    """
    # Load COVID-19 data
    covid_data = pd.read_csv(covid_data_path)
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    
    merged_data = covid_data.copy()
    
    # Merge with vaccination data if available
    if vaccination_data_path and os.path.exists(vaccination_data_path):
        vax_data = pd.read_csv(vaccination_data_path)
        vax_data['date'] = pd.to_datetime(vax_data['date'])
        
        # Select relevant columns
        vax_data = vax_data[['location', 'date', 'total_vaccinations', 'people_vaccinated', 
                             'people_fully_vaccinated', 'total_boosters']]
        
        # Rename location column to match covid_data
        vax_data = vax_data.rename(columns={'location': 'country'})
        
        # Merge on country and date
        merged_data = pd.merge(
            merged_data, 
            vax_data, 
            on=['country', 'date'], 
            how='left'
        )
    
    # Merge with mobility data if available
    if mobility_data_path and os.path.exists(mobility_data_path):
        mobility_data = pd.read_csv(mobility_data_path)
        mobility_data['date'] = pd.to_datetime(mobility_data['date'])
        
        # Select relevant columns
        mobility_data = mobility_data[['country_region', 'date', 'retail_and_recreation_percent_change_from_baseline',
                                      'grocery_and_pharmacy_percent_change_from_baseline',
                                      'parks_percent_change_from_baseline',
                                      'transit_stations_percent_change_from_baseline',
                                      'workplaces_percent_change_from_baseline',
                                      'residential_percent_change_from_baseline']]
        
        # Rename columns
        mobility_data = mobility_data.rename(columns={
            'country_region': 'country',
            'retail_and_recreation_percent_change_from_baseline': 'retail_mobility',
            'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_mobility',
            'parks_percent_change_from_baseline': 'parks_mobility',
            'transit_stations_percent_change_from_baseline': 'transit_mobility',
            'workplaces_percent_change_from_baseline': 'workplace_mobility',
            'residential_percent_change_from_baseline': 'residential_mobility'
        })
        
        # Merge on country and date
        merged_data = pd.merge(
            merged_data, 
            mobility_data, 
            on=['country', 'date'], 
            how='left'
        )
    
    # Merge with policy data if available
    if policy_data_path and os.path.exists(policy_data_path):
        policy_data = pd.read_csv(policy_data_path)
        
        # Convert date format (if needed)
        if 'Date' in policy_data.columns:
            policy_data['date'] = pd.to_datetime(policy_data['Date'], format='%Y%m%d')
        
        # Select relevant columns
        policy_data = policy_data[['CountryName', 'date', 'StringencyIndex', 'GovernmentResponseIndex',
                                  'ContainmentHealthIndex', 'EconomicSupportIndex']]
        
        # Rename columns
        policy_data = policy_data.rename(columns={
            'CountryName': 'country',
            'StringencyIndex': 'stringency_index',
            'GovernmentResponseIndex': 'govt_response_index',
            'ContainmentHealthIndex': 'containment_health_index',
            'EconomicSupportIndex': 'economic_support_index'
        })
        
        # Merge on country and date
        merged_data = pd.merge(
            merged_data, 
            policy_data, 
            on=['country', 'date'], 
            how='left'
        )
    
    return merged_data

def analyze_integrated_data(data):
    """
    Perform analysis on the integrated data to find correlations between
    COVID-19 metrics and external factors like vaccination rates and mobility.
    
    Parameters:
    data (pd.DataFrame): Integrated data with COVID-19, vaccination, mobility, and policy metrics
    
    Returns:
    dict: Dictionary with analysis results
    """
    results = {}
    
    # Check which external data is available
    vax_available = any(col in data.columns for col in ['total_vaccinations', 'people_vaccinated'])
    mobility_available = any(col in data.columns for col in ['retail_mobility', 'workplace_mobility'])
    policy_available = any(col in data.columns for col in ['stringency_index', 'govt_response_index'])
    
    # Correlation between case growth and vaccination rates
    if vax_available:
        # Calculate vaccination rate (% of population)
        if 'people_vaccinated' in data.columns and 'population' in data.columns:
            data['vaccination_rate'] = data['people_vaccinated'] / data['population'] * 100
        
        # Correlations with case growth
        vax_corr = data[['daily_cases', 'vaccination_rate']].corr()
        results['vaccination_correlations'] = vax_corr.to_dict()
    
    # Correlation between mobility changes and case growth
    if mobility_available:
        mobility_cols = [col for col in data.columns if '_mobility' in col]
        mobility_corr = data[['daily_cases'] + mobility_cols].corr()
        results['mobility_correlations'] = mobility_corr.to_dict()
    
    # Impact of policy stringency on case growth
    if policy_available:
        policy_cols = [col for col in data.columns if '_index' in col]
        policy_corr = data[['daily_cases'] + policy_cols].corr()
        results['policy_correlations'] = policy_corr.to_dict()
    
    # Calculate effectiveness metrics if all data is available
    if vax_available and policy_available:
        # Group by country and calculate averages
        country_stats = data.groupby('country').agg({
            'daily_cases': 'mean',
            'vaccination_rate': 'mean',
            'stringency_index': 'mean'
        }).reset_index()
        
        results['country_effectiveness'] = country_stats.to_dict()
    
    return results

def main():
    """Main function to execute enhanced data integration."""
    # Download additional data sources
    vaccination_data = download_vaccination_data()
    mobility_data = download_mobility_data()
    policy_data = download_policy_data()
    
    # Paths to data files
    covid_data_path = '../data/covid19_data.csv'
    vaccination_data_path = '../data/vaccination_data.csv'
    mobility_data_path = '../data/mobility_data.csv'
    policy_data_path = '../data/policy_data.csv'
    
    # Merge with COVID-19 data
    merged_data = merge_with_covid_data(
        covid_data_path, 
        vaccination_data_path,
        mobility_data_path,
        policy_data_path
    )
    
    # Save merged data
    merged_data.to_csv('../data/integrated_covid_data.csv', index=False)
    print("Integrated data saved to '../data/integrated_covid_data.csv'")
    
    # Analyze integrated data
    analysis_results = analyze_integrated_data(merged_data)
    print("Analysis results:", analysis_results)

if __name__ == "__main__":
    main() 