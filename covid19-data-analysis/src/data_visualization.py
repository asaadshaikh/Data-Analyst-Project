import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_daily_cases(df: pd.DataFrame, color: str = 'blue') -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['daily_cases'], label='Daily Cases', color=color)
    plt.title('Daily COVID-19 Cases Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_total_cases(df: pd.DataFrame, palette: str = 'viridis') -> None:
    plt.figure(figsize=(14, 7))
    sns.barplot(x='country', y='total_cases', data=df, palette=palette)
    plt.title('Total COVID-19 Cases by Country')
    plt.xlabel('Country')
    plt.ylabel('Total Cases')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot("country", "date", "daily_cases")
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".0f")
    plt.title('Heatmap of Daily COVID-19 Cases')
    plt.xlabel('Date')
    plt.ylabel('Country')
    plt.show()

def plot_case_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 8))
    plt.pie(df['total_cases'], labels=df['country'], autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Total COVID-19 Cases by Country')
    plt.axis('equal')
    plt.show()

def plot_cases_over_time(data: pd.DataFrame):
    """
    Create a line plot for COVID-19 cases over time.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing date and daily cases.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['daily_cases'], label='Daily Cases')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title('COVID-19 Daily Cases Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
