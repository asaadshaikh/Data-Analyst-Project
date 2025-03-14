import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to sys.path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="COVID-19Data Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock implementations for missing modules
def mock_load_data(file_path):
    """Mock implementation of load_data in case the module is not available"""
    return pd.read_csv(file_path)

def mock_calculate_growth_rate(data):
    """Mock implementation of calculate_growth_rate"""
    data = data.copy()
    data['growth_rate'] = data['daily_cases'].pct_change() * 100
    return data

def mock_calculate_doubling_time(data):
    """Mock implementation of calculate_doubling_time"""
    data = data.copy()
    data['doubling_time'] = np.log(2) / np.log(1 + data['growth_rate'] / 100)
    return data

def mock_fit_arima_model(data, column='daily_cases'):
    """Mock implementation of fit_arima_model"""
    class MockArimaModel:
        def forecast(self, steps=30):
            # Generate some realistic looking forecast data
            last_value = data[column].iloc[-1]
            trend = data[column].diff().mean()
            forecast = np.array([last_value + i * trend for i in range(1, steps + 1)])
            return forecast
    return MockArimaModel()

def mock_forecast_next_n_days(model, n_days=30, model_type='arima'):
    """Mock implementation of forecast_next_n_days"""
    return model.forecast(steps=n_days)

# Try to import functions from modules, use mocks if they fail
try:
    from src.data_extraction import load_data
except ImportError:
    load_data = mock_load_data
    st.warning("Using mock implementation for data_extraction")

try:
    from src.statistical_analysis import calculate_growth_rate, calculate_doubling_time
except ImportError:
    calculate_growth_rate = mock_calculate_growth_rate
    calculate_doubling_time = mock_calculate_doubling_time
    st.warning("Using mock implementation for statistical_analysis")

try:
    from src.predictive_modeling import fit_arima_model, forecast_next_n_days
except ImportError:
    fit_arima_model = mock_fit_arima_model
    forecast_next_n_days = mock_forecast_next_n_days
    st.warning("Using mock implementation for predictive_modeling")

# Dashboard title and description
st.title("COVID-19 Data Analysis Dashboard")
st.markdown("""
This dashboard provides interactive visualizations and analysis of COVID-19 data.
Select different countries, date ranges, and metrics to explore the pandemic's impact.
""")

# Load and prepare data
def load_data():
    file_path = '../data/covid19_data.csv'
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataframe if file is not found
        dates = pd.date_range(start='2020-01-22', end='2023-04-30')
        countries = ['US', 'India', 'Brazil', 'UK', 'Russia', 'France', 'Germany', 'Italy', 'Spain', 'China']
        
        # Create sample data
        sample_data = []
        for country in countries:
            cases_scale = np.random.randint(1000, 50000)
            deaths_scale = np.random.randint(10, 1000)
            recovered_scale = np.random.randint(500, 30000)
            
            for date in dates:
                day_num = (date - dates[0]).days
                # Create some realistic looking trends with randomness
                cases = int(cases_scale * (1 + np.sin(day_num/30)) * (1 + day_num/100) * np.random.uniform(0.8, 1.2))
                deaths = int(deaths_scale * (1 + np.sin(day_num/30)) * (1 + day_num/150) * np.random.uniform(0.8, 1.2))
                recovered = int(recovered_scale * (1 + np.sin(day_num/30)) * (1 + day_num/120) * np.random.uniform(0.8, 1.2))
                
                sample_data.append({
                    'date': date,
                    'country': country,
                    'confirmed_cases': cases,
                    'deaths': deaths,
                    'recovered': recovered,
                    'active_cases': cases - deaths - recovered
                })
        
        df = pd.DataFrame(sample_data)
        return df

# Load data
data = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
min_date = data['date'].min().date()
max_date = data['date'].max().date()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Convert to datetime for filtering
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Country filter
all_countries = sorted(data['country'].unique())
selected_countries = st.sidebar.multiselect("Select Countries", all_countries, default=all_countries[:5])

# Metric filter
metrics = ["confirmed_cases", "deaths", "recovered", "active_cases"]
selected_metric = st.sidebar.selectbox("Select Metric", metrics)

# Filter data based on selections
filtered_data = data[
    (data['date'] >= start_date) & 
    (data['date'] <= end_date) & 
    (data['country'].isin(selected_countries))
]

# Dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Geographic Analysis", "Country Comparison", "Predictions"])

# Tab 1: Overview
with tab1:
    st.header("COVID-19 Overview")
    
    # Summary metrics
    if not filtered_data.empty:
        latest_date = filtered_data['date'].max()
        latest_data = filtered_data[filtered_data['date'] == latest_date]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cases = filtered_data['confirmed_cases'].sum()
            st.metric("Total Cases", f"{total_cases:,}")
            
        with col2:
            total_deaths = filtered_data['deaths'].sum()
            st.metric("Total Deaths", f"{total_deaths:,}")
            
        with col3:
            total_recovered = filtered_data['recovered'].sum()
            st.metric("Total Recovered", f"{total_recovered:,}")
            
        with col4:
            total_active = filtered_data['active_cases'].sum()
            st.metric("Active Cases", f"{total_active:,}")
        
        # Daily trends
        st.subheader(f"Daily {selected_metric.replace('_', ' ').title()} Trend")
        daily_data = filtered_data.groupby('date')[selected_metric].sum().reset_index()
        
        fig = px.line(
            daily_data, 
            x='date', 
            y=selected_metric,
            labels={
                'date': 'Date',
                selected_metric: selected_metric.replace('_', ' ').title()
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(filtered_data)
    else:
        st.info("No data available for the selected filters. Please adjust your selection.")

with tab2:
    st.header("Geographic Analysis")
    
    # World map visualization
    if not filtered_data.empty:
        # Aggregate data by country
        country_totals = filtered_data.groupby('country')[selected_metric].sum().reset_index()
        
        fig = px.choropleth(
            country_totals,
            locations="country",
            locationmode="country names",
            color=selected_metric,
            hover_name="country",
            color_continuous_scale="Viridis",
            title=f"{selected_metric.replace('_', ' ').title()} by Country",
            labels={selected_metric: selected_metric.replace('_', ' ').title()}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top affected countries
        st.subheader(f"Top 10 Countries by {selected_metric.replace('_', ' ').title()}")
        top_countries = country_totals.sort_values(selected_metric, ascending=False).head(10)
        
        fig = px.bar(
            top_countries,
            x="country",
            y=selected_metric,
            labels={
                "country": "Country",
                selected_metric: selected_metric.replace('_', ' ').title()
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please adjust your selection.")

with tab3:
    st.header("Country Comparison")
    
    # Bar chart comparison
    if len(selected_countries) > 0:
        country_comparison = filtered_data.groupby('country')[selected_metric].sum().reset_index()
        country_comparison = country_comparison.sort_values(selected_metric, ascending=False)
        
        fig = px.bar(
            country_comparison,
            x="country",
            y=selected_metric,
            title=f"{selected_metric.replace('_', ' ').title()} by Country",
            labels={"country": "Country", selected_metric: selected_metric.replace('_', ' ').title()}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Line chart comparison
        country_time_series = filtered_data.groupby(['country', 'date'])[selected_metric].sum().reset_index()
        fig = px.line(
            country_time_series,
            x="date",
            y=selected_metric,
            color="country",
            title=f"{selected_metric.replace('_', ' ').title()} Trends by Country",
            labels={
                "date": "Date", 
                selected_metric: selected_metric.replace('_', ' ').title(),
                "country": "Country"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one country to see comparison")

with tab4:
    st.header("Predictions")
    
    if not filtered_data.empty and len(selected_countries) > 0:
        # Select a single country for prediction
        prediction_country = st.selectbox(
            "Select Country for Prediction", 
            options=selected_countries,
            index=0 if selected_countries else None
        )
        
        if prediction_country:
            # Filter data for the selected country
            country_data = filtered_data[filtered_data['country'] == prediction_country]
            time_series = country_data.groupby('date')[selected_metric].sum().reset_index()
            
            # Number of days to forecast
            forecast_days = st.slider("Days to Forecast", min_value=7, max_value=30, value=14)
            
            # Simple prediction (using moving average)
            st.subheader(f"{selected_metric.replace('_', ' ').title()} Forecast for {prediction_country}")
            
            if len(time_series) >= 7:  # Need at least 7 days of data
                # Calculate moving average for last 7 days
                last_value = time_series[selected_metric].iloc[-1]
                avg_change = time_series[selected_metric].diff().iloc[-7:].mean()
                
                # Create forecast dates
                last_date = time_series['date'].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                
                # Create forecast values (linear trend + some noise)
                forecast_values = [last_value + avg_change * (i+1) + np.random.normal(0, abs(avg_change*0.1)) for i in range(forecast_days)]
                forecast_values = [max(0, v) for v in forecast_values]  # Ensure no negative values
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    selected_metric: forecast_values,
                    'type': 'forecast'
                })
                
                # Add type to historical data
                time_series['type'] = 'historical'
                
                # Combine historical and forecast data
                combined_df = pd.concat([time_series, forecast_df])
                
                # Plot the results
                fig = px.line(
                    combined_df,
                    x='date',
                    y=selected_metric,
                    color='type',
                    title=f"{selected_metric.replace('_', ' ').title()} Forecast for {prediction_country}",
                    labels={
                        'date': 'Date',
                        selected_metric: selected_metric.replace('_', ' ').title(),
                        'type': 'Data Type'
                    },
                    color_discrete_map={'historical': 'blue', 'forecast': 'red'}
                )
                
                # Add a vertical line at the forecast start date
                fig.add_vline(x=last_date, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast values
                st.subheader("Forecast Values")
                st.dataframe(forecast_df[['date', selected_metric]])
                
                # Add disclaimer
                st.info("Disclaimer: This is a simple forecast based on recent trends and should not be used for critical decision-making.")
            else:
                st.warning(f"Not enough data available for {prediction_country} to generate a forecast. Please select a different country or adjust your date range.")
    else:
        st.info("Please select at least one country to see predictions.")

# Footer
st.markdown("---")
st.markdown("COVID-19 Data Analysis Dashboard | Created with Streamlit")

if __name__ == "__main__":
    # Run with: streamlit run dashboard.py
    pass
