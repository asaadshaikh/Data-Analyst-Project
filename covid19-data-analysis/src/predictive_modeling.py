import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path):
    """Load the COVID-19 dataset from a CSV file."""
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    return data

def prepare_time_series_data(data, target_col='daily_cases', n_days=14):
    """
    Prepare time series data for forecasting by creating lagged features.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing 'date' and target_col
    target_col (str): Column to forecast
    n_days (int): Number of days to use for lagged features
    
    Returns:
    pd.DataFrame: DataFrame with features and target for ML models
    """
    # Sort by date
    data = data.sort_values('date')
    
    # Create lagged features
    df_features = pd.DataFrame()
    if target_col not in data.columns:
        raise ValueError(f"The DataFrame must contain a '{target_col}' column.")
    df_features['target'] = data[target_col]

    
    for lag in range(1, n_days + 1):
        df_features[f'lag_{lag}'] = data[target_col].shift(lag)
    
    # Add day of week and month as features
    df_features['day_of_week'] = data['date'].dt.dayofweek
    df_features['month'] = data['date'].dt.month
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

def train_linear_regression(features, target):
    """Train a linear regression model for forecasting."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics

def train_random_forest(features, target):
    """Train a random forest model for forecasting."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics

def fit_arima_model(data, column='daily_cases', order=(5,1,0)):
    """
    Fit an ARIMA model for time series forecasting.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the time series data
    column (str): Column to forecast
    order (tuple): ARIMA model order (p,d,q)
    
    Returns:
    model: Fitted ARIMA model
    """
    # Prepare the data for ARIMA
    if series.empty:
        raise ValueError("The series for ARIMA fitting is empty.")
    series = data[column]

    
    # Fit the ARIMA model
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    return model_fit

def forecast_next_n_days(model, n_days=30, model_type='arima'):
    """
    Forecast COVID-19 cases for the next n days.
    
    Parameters:
    model: Trained model
    n_days (int): Number of days to forecast
    model_type (str): Type of model ('arima', 'linear', 'rf')
    
    Returns:
    array: Forecasted values
    """
    if model_type == 'arima':
        forecast = model.forecast(steps=n_days)
        return forecast
    # For ML models, we would need the latest data and features to predict iteratively
    # This would be more complex to implement here
    
def plot_forecast(actual_data, forecasted_data, title='COVID-19 Case Forecast'):
    """Plot actual vs forecasted data."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data.index, actual_data, label='Actual', color='blue')
    plt.plot(forecasted_data.index, forecasted_data, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to execute predictive modeling."""
    file_path = '../data/covid19_data.csv'
    data = load_data(file_path)
    
    # Example for one country/region
    country_data = data[data['country'] == 'US'].copy()
    
    # Prepare features for ML models
    df_features = prepare_time_series_data(country_data)
    features = df_features.drop('target', axis=1)
    target = df_features['target']
    
    # Train and evaluate models
    lr_model, lr_metrics = train_linear_regression(features, target)
    rf_model, rf_metrics = train_random_forest(features, target)
    
    print("Linear Regression Metrics:", lr_metrics)
    print("Random Forest Metrics:", rf_metrics)
    
    # ARIMA modeling
    arima_model = fit_arima_model(country_data)
    
    # Forecast next 30 days
    forecast = forecast_next_n_days(arima_model, n_days=30)
    print("ARIMA Forecast for next 30 days:", forecast)

if __name__ == "__main__":
    main()
