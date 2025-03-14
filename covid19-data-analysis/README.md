# COVID-19 Data Analysis and Visualization

## Objective
The objective of this project is to analyze COVID-19 data to gain insights into trends and patterns, and to visualize the data using various charts and graphs. This project also includes data extraction, cleaning, and statistical analysis to provide comprehensive insights.

## Installation
To set up the project, clone the repository and install the required packages using the following command:
```
pip install -r requirements.txt
```

## Usage
### Basic Analysis
Run the main script to execute the data analysis:
```
python src/exploratory_data_analysis.py
```

### Interactive Dashboard
Launch the interactive dashboard:
```
streamlit run src/dashboard.py
```

### Predictive Modeling
Run the predictive modeling script:
```
python src/predictive_modeling.py
```

### Epidemiological Models
Run the epidemiological models:
```
python src/epidemiological_models.py
```

## Skills Demonstrated
- Data extraction and cleaning
- Data visualization and statistical analysis
- Machine learning and time series forecasting
- Interactive dashboard development
- Epidemiological modeling (SIR/SEIR models)
- Performance optimization techniques
- Integration of multiple data sources

## Tools/Technologies
- Python
- Pandas/NumPy for data manipulation
- Matplotlib/Seaborn for visualization
- Plotly for interactive visualizations
- Streamlit for dashboard development
- Scikit-learn for machine learning
- SciPy for differential equation solving
- Dask for parallel processing
- Joblib for caching optimization

## Project Structure
```
covid19-data-analysis
├── data
│   ├── covid19_data.csv          # COVID-19 dataset
│   ├── vaccination_data.csv      # Vaccination data
│   ├── mobility_data.csv         # Mobility data
│   └── policy_data.csv           # Policy intervention data
├── notebooks
│   ├── data_extraction.ipynb     # Data extraction notebook
│   ├── data_cleaning.ipynb       # Data cleaning notebook
│   ├── exploratory_data_analysis.ipynb # EDA notebook
│   ├── data_visualization.ipynb   # Data visualization notebook
│   └── statistical_analysis.ipynb  # Statistical analysis notebook
├── src
│   ├── data_extraction.py         # Data extraction script
│   ├── data_cleaning.py           # Data cleaning script
│   ├── exploratory_data_analysis.py # EDA script
│   ├── data_visualization.py       # Visualization script
│   ├── statistical_analysis.py      # Statistical analysis script
│   ├── predictive_modeling.py      # ML forecasting script
│   ├── dashboard.py                # Interactive Streamlit dashboard
│   ├── enhanced_data_sources.py    # Additional data sources integration
│   ├── performance_optimization.py # Performance optimization utilities
│   └── epidemiological_models.py   # SIR/SEIR models implementation
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Key Features

### 1. Machine Learning Forecasting
The project includes time series forecasting models for predicting future COVID-19 cases:
- ARIMA models for time series forecasting
- Linear regression for trend analysis
- Random forest for complex pattern recognition

### 2. Interactive Dashboard
The Streamlit dashboard provides an interactive interface to explore COVID-19 data:
- Geographic visualizations using choropleth maps
- Time series trend analysis with customizable date ranges
- Country comparison tools
- Prediction visualization

### 3. Enhanced Data Sources
The project integrates multiple data sources for comprehensive analysis:
- Vaccination data from Our World in Data
- Mobility data from Google Community Mobility Reports
- Policy intervention data from Oxford COVID-19 Government Response Tracker

### 4. Performance Optimization
The project includes performance optimization techniques:
- Data type optimization to reduce memory usage
- Parallel processing for faster data analysis
- Caching mechanism for improved loading times
- Efficient data formats (Parquet) for storage

### 5. Epidemiological Models
The project implements mathematical models for disease spread:
- SIR (Susceptible-Infected-Recovered) model
- SEIR (Susceptible-Exposed-Infected-Recovered) model
- Parameter fitting to actual COVID-19 data
- Comparative analysis between models

## Instructions
1. **Data Extraction**: Use the `data_extraction.ipynb` notebook to load the COVID-19 dataset from the `data/covid19_data.csv` file. Ensure the URL in the `data_extraction.py` file points to a valid dataset.
2. **Data Cleaning**: Clean the dataset using the `data_cleaning.ipynb` notebook to handle missing values and standardize formats. The `data_cleaning.py` script should be verified to ensure it correctly loads the dataset.
3. **Exploratory Data Analysis**: Perform EDA using the `exploratory_data_analysis.ipynb` notebook to calculate statistics and identify trends. Ensure the data is cleaned before performing EDA.
4. **Data Visualization**: Create visualizations in the `data_visualization.ipynb` notebook to represent the data graphically. Verify that the data passed to the visualization functions is correctly formatted.
5. **Statistical Analysis**: Conduct statistical analysis using the `statistical_analysis.ipynb` notebook to explore relationships and growth rates. The `statistical_analysis.py` script should be checked for any syntax errors, particularly in the doubling time calculation.
6. **Enhanced Data Integration**: Run the `enhanced_data_sources.py` script to download and integrate additional data sources like vaccination rates, mobility data, and policy interventions.
7. **Interactive Dashboard**: Launch the Streamlit dashboard with `streamlit run src/dashboard.py` to explore the data interactively.
8. **Forecasting**: Use the `predictive_modeling.py` script to forecast future COVID-19 cases using machine learning models.
9. **Epidemiological Modeling**: Apply the SIR and SEIR models using the `epidemiological_models.py` script to understand the dynamics of disease spread.

## Visualization Examples
Here are some examples of visualizations created in this project:

### Example 1: Daily New Cases
![Daily New Cases](images/daily_new_cases.png)

### Example 2: Cumulative Cases Over Time
![Cumulative Cases Over Time](images/cumulative_cases.png)

### Example 3: Death Rate by Country
![Death Rate by Country](images/death_rate_by_country.png)

### Example 4: Interactive Dashboard
![Interactive Dashboard](images/dashboard.png)

### Example 5: SIR/SEIR Model Comparison
![Epidemiological Models](images/epidemiological_models.png)

## Conclusion
This project provides a comprehensive analysis of COVID-19 data, offering insights and visualizations that help understand the pandemic's impact. The enhanced features provide deeper analysis capabilities, predictive insights, and interactive exploration tools.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
