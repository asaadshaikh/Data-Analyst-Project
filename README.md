# COVID-19 Data Analysis and Visualization

## Objective
The objective of this project is to analyze COVID-19 data to gain insights into trends and patterns, and to visualize the data using various charts and graphs. This project also includes data extraction, cleaning, and statistical analysis to provide comprehensive insights.

## Installation
To set up the project, clone the repository and install the required packages using the following command:
```
pip install -r requirements.txt
```

## Usage
Run the main script to execute the data analysis:
```
python src/exploratory_data_analysis.py
```

## Skills Demonstrated
- Data extraction
- Data cleaning
- Data visualization
- Statistical analysis

## Tools/Technologies
- Python
- Pandas
- Matplotlib
- Seaborn

## Project Structure
```
covid19-data-analysis
├── data
│   └── covid19_data.csv          # COVID-19 dataset
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
│   └── statistical_analysis.py      # Statistical analysis script
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Methods Used
### Data Extraction
We use various APIs and web scraping techniques to extract COVID-19 data from reliable sources. The `extract_data` function handles the data extraction process, ensuring that the data is up-to-date and accurate.

### Data Cleaning
The `clean_data` function is responsible for cleaning the raw data. This includes handling missing values, correcting data formats, and removing any inconsistencies to ensure the data is ready for analysis.

### Data Analysis
The `analyze_data` function performs various statistical analyses on the cleaned data. This includes calculating key metrics such as infection rates, recovery rates, and mortality rates. We also use time series analysis to track the progression of the pandemic over time.

### Data Visualization
The `visualize_data` function generates visual representations of the analysis results. We use libraries like Matplotlib and Seaborn to create static plots, and Plotly for interactive visualizations. These visualizations help in understanding the trends and patterns in the data.

## Significance of Results
The results of our analysis provide valuable insights into the COVID-19 pandemic. By understanding the trends and patterns in the data, we can make informed decisions about public health measures and resource allocation. Our visualizations make it easier to communicate these insights to a broader audience.

## Instructions
1. **Data Extraction**: Use the `data_extraction.ipynb` notebook to load the COVID-19 dataset from the `data/covid19_data.csv` file. Ensure the URL in the `data_extraction.py` file points to a valid dataset.
2. **Data Cleaning**: Clean the dataset using the `data_cleaning.ipynb` notebook to handle missing values and standardize formats. The `data_cleaning.py` script should be verified to ensure it correctly loads the dataset.
3. **Exploratory Data Analysis**: Perform EDA using the `exploratory_data_analysis.ipynb` notebook to calculate statistics and identify trends. Ensure the data is cleaned before performing EDA.
4. **Data Visualization**: Create visualizations in the `data_visualization.ipynb` notebook to represent the data graphically. Verify that the data passed to the visualization functions is correctly formatted.
5. **Statistical Analysis**: Conduct statistical analysis using the `statistical_analysis.ipynb` notebook to explore relationships and growth rates. The `statistical_analysis.py` script should be checked for any syntax errors, particularly in the doubling time calculation.

## Conclusion
This project aims to provide a comprehensive analysis of COVID-19 data, offering insights and visualizations that can help in understanding the pandemic's impact.
