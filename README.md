# COVID-19 Data Analysis Dashboard

An interactive dashboard for analyzing and visualizing COVID-19 data across different countries. This project provides insights into the pandemic's impact through various visualization tools and predictive analytics.

## Features

- **Interactive Dashboard**: Built with Streamlit for an intuitive user interface
- **Multiple Visualization Types**: Line charts, bar charts, and choropleth maps
- **Geographic Analysis**: World map visualization of COVID-19 impact
- **Country Comparison**: Tools to compare metrics across multiple countries
- **Predictive Analytics**: Simple forecasting capabilities
- **Data Filtering**: Filter by countries, date ranges, and different metrics
- **Error Handling**: Robust error management with graceful fallbacks

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid19-data-analysis.git
cd covid19-data-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:
```bash
streamlit run src/dashboard.py
```

The dashboard will be available at http://localhost:8501

## Project Structure

```
covid19-data-analysis/
├── data/                  # Data files
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── dashboard.py     # Main dashboard application
│   ├── data_extraction.py
│   ├── statistical_analysis.py
│   └── predictive_modeling.py
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Future Improvements

- Implement advanced ML models for more accurate forecasting
- Integrate with real-time data APIs
- Add vaccination and testing data
- Develop a database backend
- Implement user authentication
- Create API endpoints
- Optimize for mobile devices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
