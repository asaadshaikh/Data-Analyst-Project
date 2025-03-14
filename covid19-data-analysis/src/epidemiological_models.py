import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('epidemiological_models')

class SIRModel:
    """
    SIR (Susceptible-Infected-Recovered) model for epidemic modeling.
    
    The SIR model describes the changes in the number of people in each of 
    three compartments: S (susceptible), I (infected), and R (recovered/removed).
    """
    
    def __init__(self, beta=0.3, gamma=0.1):
        """
        Initialize the SIR model with transmission and recovery rates.
        
        Parameters:
        beta (float): Transmission rate (rate at which susceptible individuals become infected)
        gamma (float): Recovery rate (rate at which infected individuals recover)
        """
        self.beta = beta
        self.gamma = gamma
        
    def model(self, t, y, population):
        """
        SIR model differential equations.
        
        Parameters:
        t (float): Time point
        y (array): Current state [S, I, R]
        population (int): Total population
        
        Returns:
        array: Derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        
        # Ensure values don't go below 0 or above population
        S = max(0, min(S, population))
        I = max(0, min(I, population))
        R = max(0, min(R, population))
        
        # Calculate derivatives
        dSdt = -self.beta * S * I / population
        dIdt = self.beta * S * I / population - self.gamma * I
        dRdt = self.gamma * I
        
        return [dSdt, dIdt, dRdt]
    
    def simulate(self, population, initial_infected, initial_recovered, days):
        """
        Simulate the SIR model for a given number of days.
        
        Parameters:
        population (int): Total population
        initial_infected (int): Initial number of infected individuals
        initial_recovered (int): Initial number of recovered individuals
        days (int): Number of days to simulate
        
        Returns:
        dict: Dictionary with time points and S, I, R values
        """
        # Calculate initial susceptible
        initial_susceptible = population - initial_infected - initial_recovered
        
        # Initial conditions
        y0 = [initial_susceptible, initial_infected, initial_recovered]
        
        # Time points
        t = np.linspace(0, days, days + 1)
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.model(t, y, population),
            [0, days],
            y0,
            t_eval=t,
            method='RK45'
        )
        
        # Extract results
        S, I, R = sol.y
        
        return {
            'time': t,
            'susceptible': S,
            'infected': I,
            'recovered': R
        }
    
    def fit_to_data(self, actual_cases, population, initial_infected, initial_recovered, days):
        """
        Fit the SIR model to actual case data by optimizing beta and gamma.
        
        Parameters:
        actual_cases (array): Array of actual cumulative cases
        population (int): Total population
        initial_infected (int): Initial number of infected individuals
        initial_recovered (int): Initial number of recovered individuals
        days (int): Number of days to simulate
        
        Returns:
        tuple: Optimized beta and gamma values
        """
        # Define objective function to minimize
        def objective(params):
            beta, gamma = params
            self.beta = beta
            self.gamma = gamma
            
            sim_results = self.simulate(population, initial_infected, initial_recovered, days)
            
            # Calculate cumulative cases (infected + recovered)
            simulated_cases = sim_results['infected'] + sim_results['recovered']
            
            # Calculate mean squared error
            mse = np.mean((simulated_cases[:len(actual_cases)] - actual_cases) ** 2)
            
            return mse
        
        # Initial parameter guess
        initial_guess = [0.3, 0.1]  # beta, gamma
        
        # Parameter bounds
        bounds = [(0.01, 1.0), (0.01, 1.0)]  # beta, gamma
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update model parameters with optimized values
        self.beta, self.gamma = result.x
        
        return self.beta, self.gamma
    
    def plot(self, simulation_results, title="SIR Model Simulation"):
        """
        Plot the SIR model simulation results.
        
        Parameters:
        simulation_results (dict): Results from the simulate method
        title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        plt.plot(simulation_results['time'], simulation_results['susceptible'], label='Susceptible')
        plt.plot(simulation_results['time'], simulation_results['infected'], label='Infected')
        plt.plot(simulation_results['time'], simulation_results['recovered'], label='Recovered')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def interactive_plot(self, simulation_results, title="SIR Model Simulation"):
        """
        Create an interactive plot of the SIR model using Plotly.
        
        Parameters:
        simulation_results (dict): Results from the simulate method
        title (str): Plot title
        
        Returns:
        plotly.graph_objects.Figure: Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['susceptible'],
            mode='lines',
            name='Susceptible',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['infected'],
            mode='lines',
            name='Infected',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['recovered'],
            mode='lines',
            name='Recovered',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Days',
            yaxis_title='Population',
            legend_title='Compartments',
            hovermode='x unified'
        )
        
        return fig


class SEIRModel:
    """
    SEIR (Susceptible-Exposed-Infected-Recovered) model for epidemic modeling.
    
    The SEIR model extends the SIR model by adding an 'Exposed' compartment for
    individuals who have been infected but are not yet infectious.
    """
    
    def __init__(self, beta=0.3, sigma=0.2, gamma=0.1):
        """
        Initialize the SEIR model with transmission, incubation, and recovery rates.
        
        Parameters:
        beta (float): Transmission rate
        sigma (float): Rate at which exposed individuals become infectious (1/incubation period)
        gamma (float): Recovery rate
        """
        self.beta = beta
        self.sigma = sigma  # Incubation rate
        self.gamma = gamma
        
    def model(self, t, y, population):
        """
        SEIR model differential equations.
        
        Parameters:
        t (float): Time point
        y (array): Current state [S, E, I, R]
        population (int): Total population
        
        Returns:
        array: Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        
        # Ensure values don't go below 0 or above population
        S = max(0, min(S, population))
        E = max(0, min(E, population))
        I = max(0, min(I, population))
        R = max(0, min(R, population))
        
        # Calculate derivatives
        dSdt = -self.beta * S * I / population
        dEdt = self.beta * S * I / population - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        
        return [dSdt, dEdt, dIdt, dRdt]
    
    def simulate(self, population, initial_exposed, initial_infected, initial_recovered, days):
        """
        Simulate the SEIR model for a given number of days.
        
        Parameters:
        population (int): Total population
        initial_exposed (int): Initial number of exposed individuals
        initial_infected (int): Initial number of infected individuals
        initial_recovered (int): Initial number of recovered individuals
        days (int): Number of days to simulate
        
        Returns:
        dict: Dictionary with time points and S, E, I, R values
        """
        # Calculate initial susceptible
        initial_susceptible = population - initial_exposed - initial_infected - initial_recovered
        
        # Initial conditions
        y0 = [initial_susceptible, initial_exposed, initial_infected, initial_recovered]
        
        # Time points
        t = np.linspace(0, days, days + 1)
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.model(t, y, population),
            [0, days],
            y0,
            t_eval=t,
            method='RK45'
        )
        
        # Extract results
        S, E, I, R = sol.y
        
        return {
            'time': t,
            'susceptible': S,
            'exposed': E,
            'infected': I,
            'recovered': R
        }
    
    def fit_to_data(self, actual_cases, population, initial_exposed, initial_infected, initial_recovered, days):
        """
        Fit the SEIR model to actual case data by optimizing beta, sigma, and gamma.
        
        Parameters:
        actual_cases (array): Array of actual cumulative cases
        population (int): Total population
        initial_exposed (int): Initial number of exposed individuals
        initial_infected (int): Initial number of infected individuals
        initial_recovered (int): Initial number of recovered individuals
        days (int): Number of days to simulate
        
        Returns:
        tuple: Optimized beta, sigma, and gamma values
        """
        # Define objective function to minimize
        def objective(params):
            beta, sigma, gamma = params
            self.beta = beta
            self.sigma = sigma
            self.gamma = gamma
            
            sim_results = self.simulate(population, initial_exposed, initial_infected, initial_recovered, days)
            
            # Calculate cumulative cases (exposed + infected + recovered)
            simulated_cases = sim_results['exposed'] + sim_results['infected'] + sim_results['recovered']
            
            # Calculate mean squared error
            mse = np.mean((simulated_cases[:len(actual_cases)] - actual_cases) ** 2)
            
            return mse
        
        # Initial parameter guess
        initial_guess = [0.3, 0.2, 0.1]  # beta, sigma, gamma
        
        # Parameter bounds
        bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 1.0)]  # beta, sigma, gamma
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update model parameters with optimized values
        self.beta, self.sigma, self.gamma = result.x
        
        return self.beta, self.sigma, self.gamma
    
    def plot(self, simulation_results, title="SEIR Model Simulation"):
        """
        Plot the SEIR model simulation results.
        
        Parameters:
        simulation_results (dict): Results from the simulate method
        title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        plt.plot(simulation_results['time'], simulation_results['susceptible'], label='Susceptible')
        plt.plot(simulation_results['time'], simulation_results['exposed'], label='Exposed')
        plt.plot(simulation_results['time'], simulation_results['infected'], label='Infected')
        plt.plot(simulation_results['time'], simulation_results['recovered'], label='Recovered')
        plt.xlabel('Days')
        plt.ylabel('Population')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def interactive_plot(self, simulation_results, title="SEIR Model Simulation"):
        """
        Create an interactive plot of the SEIR model using Plotly.
        
        Parameters:
        simulation_results (dict): Results from the simulate method
        title (str): Plot title
        
        Returns:
        plotly.graph_objects.Figure: Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['susceptible'],
            mode='lines',
            name='Susceptible',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['exposed'],
            mode='lines',
            name='Exposed',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['infected'],
            mode='lines',
            name='Infected',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=simulation_results['time'],
            y=simulation_results['recovered'],
            mode='lines',
            name='Recovered',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Days',
            yaxis_title='Population',
            legend_title='Compartments',
            hovermode='x unified'
        )
        
        return fig

def apply_model_to_country_data(covid_data, country, population, model_type='SIR'):
    """
    Apply an epidemiological model to actual country data.
    
    Parameters:
    covid_data (pd.DataFrame): DataFrame containing COVID-19 data
    country (str): Country name
    population (int): Country population
    model_type (str): Type of model to use ('SIR' or 'SEIR')
    
    Returns:
    tuple: (Fitted model, Simulation results, Actual cases)
    """
    # Filter data for the specified country
    country_data = covid_data[covid_data['country'] == country].copy()
    
    # Ensure data is sorted by date
    country_data = country_data.sort_values('date')
    
    # Get actual case data
    actual_cases = country_data['total_cases'].values
    
    # Get number of days
    days = len(actual_cases)
    
    # Initial values
    initial_infected = actual_cases[0] if len(actual_cases) > 0 else 1
    initial_recovered = 0  # Assuming no recovered cases at the start
    
    # Create and fit model
    if model_type == 'SIR':
        model = SIRModel()
        
        # Fit model to data
        beta, gamma = model.fit_to_data(
            actual_cases,
            population,
            initial_infected,
            initial_recovered,
            days
        )
        
        logger.info(f"Fitted SIR model for {country}: beta={beta:.4f}, gamma={gamma:.4f}")
        
        # Simulate with fitted parameters
        simulation_results = model.simulate(
            population,
            initial_infected,
            initial_recovered,
            days
        )
        
    elif model_type == 'SEIR':
        model = SEIRModel()
        
        # Initial exposed individuals (assumed to be 2x initial infected)
        initial_exposed = 2 * initial_infected
        
        # Fit model to data
        beta, sigma, gamma = model.fit_to_data(
            actual_cases,
            population,
            initial_exposed,
            initial_infected,
            initial_recovered,
            days
        )
        
        logger.info(f"Fitted SEIR model for {country}: beta={beta:.4f}, sigma={sigma:.4f}, gamma={gamma:.4f}")
        
        # Simulate with fitted parameters
        simulation_results = model.simulate(
            population,
            initial_exposed,
            initial_infected,
            initial_recovered,
            days
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, simulation_results, actual_cases

def compare_models_with_actual_data(covid_data, country, population):
    """
    Compare SIR and SEIR models with actual data for a country.
    
    Parameters:
    covid_data (pd.DataFrame): DataFrame containing COVID-19 data
    country (str): Country name
    population (int): Country population
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure comparing models and actual data
    """
    # Apply models
    sir_model, sir_results, actual_cases = apply_model_to_country_data(
        covid_data, country, population, model_type='SIR'
    )
    
    seir_model, seir_results, _ = apply_model_to_country_data(
        covid_data, country, population, model_type='SEIR'
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=list(range(len(actual_cases))),
        y=actual_cases,
        mode='markers',
        name='Actual Cases',
        marker=dict(color='black', size=8)
    ))
    
    # Add SIR model predictions
    sir_cases = sir_results['infected'] + sir_results['recovered']
    fig.add_trace(go.Scatter(
        x=list(range(len(sir_cases))),
        y=sir_cases,
        mode='lines',
        name='SIR Model',
        line=dict(color='blue', width=2)
    ))
    
    # Add SEIR model predictions
    seir_cases = seir_results['exposed'] + seir_results['infected'] + seir_results['recovered']
    fig.add_trace(go.Scatter(
        x=list(range(len(seir_cases))),
        y=seir_cases,
        mode='lines',
        name='SEIR Model',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title=f"COVID-19 Cases in {country}: Actual vs Model Predictions",
        xaxis_title='Days since first case',
        yaxis_title='Cumulative Cases',
        legend_title='Data Source',
        hovermode='x unified'
    )
    
    return fig

def main():
    """Main function to demonstrate epidemiological models."""
    # Load COVID-19 data
    covid_data_path = '../data/covid19_data.csv'
    
    try:
        covid_data = pd.read_csv(covid_data_path)
        covid_data['date'] = pd.to_datetime(covid_data['date'])
        
        # Example: Apply SIR model to US data
        us_population = 331000000  # US population estimate
        
        sir_model, sir_results, actual_cases = apply_model_to_country_data(
            covid_data, 'US', us_population, model_type='SIR'
        )
        
        # Plot results
        sir_model.plot(sir_results, title="SIR Model for US COVID-19 Cases")
        
        # Example: Apply SEIR model to US data
        seir_model, seir_results, _ = apply_model_to_country_data(
            covid_data, 'US', us_population, model_type='SEIR'
        )
        
        # Plot results
        seir_model.plot(seir_results, title="SEIR Model for US COVID-19 Cases")
        
        # Compare models
        fig = compare_models_with_actual_data(covid_data, 'US', us_population)
        
        logger.info("Epidemiological models successfully applied to data.")
        
    except Exception as e:
        logger.error(f"Error applying epidemiological models: {e}")

if __name__ == "__main__":
    main() 