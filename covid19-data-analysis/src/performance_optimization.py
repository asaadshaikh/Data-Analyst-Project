import pandas as pd
import numpy as np
import time
import os
import multiprocessing
from functools import partial
import dask.dataframe as dd
import psutil
import joblib
from concurrent.futures import ProcessPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_optimization')

def memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def benchmark(func):
    """Decorator to benchmark function execution time and memory usage."""
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start_mem = memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = memory_usage()
        
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {end_mem - start_mem:.2f} MB")
        
        return result
    
    return wrapper

def optimize_dataframe(df):
    """
    Optimize DataFrame memory usage by converting to appropriate data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to optimize
    
    Returns:
    pd.DataFrame: Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / (1024 * 1024)
    logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
    
    # Convert numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                df[col] = df[col].astype('int32')
    
    # Convert float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Convert object columns to categories if appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < df.shape[0] * 0.5:  # If cardinality is less than 50% of rows
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / (1024 * 1024)
    logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
    logger.info(f"Memory usage reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df

def process_chunk(chunk, operation_func):
    """
    Process a chunk of data with the given operation function.
    
    Parameters:
    chunk (pd.DataFrame): DataFrame chunk to process
    operation_func (function): Function to apply to the chunk
    
    Returns:
    pd.DataFrame: Processed chunk
    """
    return operation_func(chunk)

@benchmark
def parallel_process_dataframe(df, operation_func, num_partitions=None):
    """
    Process DataFrame in parallel using multiple cores.
    
    Parameters:
    df (pd.DataFrame): DataFrame to process
    operation_func (function): Function to apply to each chunk
    num_partitions (int): Number of partitions to split the DataFrame into
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    if num_partitions is None:
        num_partitions = multiprocessing.cpu_count()
    
    # Split DataFrame into chunks
    chunks = np.array_split(df, num_partitions)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_partitions) as executor:
        results = list(executor.map(partial(process_chunk, operation_func=operation_func), chunks))
    
    # Combine results
    return pd.concat(results)

@benchmark
def dask_process_dataframe(df, operation_func, num_partitions=None):
    """
    Process DataFrame using Dask for out-of-core computation.
    
    Parameters:
    df (pd.DataFrame): DataFrame to process
    operation_func (function): Function to apply to the DataFrame
    num_partitions (int): Number of partitions for Dask
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    if num_partitions is None:
        num_partitions = multiprocessing.cpu_count()
    
    # Convert to Dask DataFrame
    dask_df = dd.from_pandas(df, npartitions=num_partitions)
    
    # Apply operation
    result = dask_df.map_partitions(operation_func)
    
    # Compute and return result
    return result.compute()

def optimize_pandas_operations(df):
    """
    Apply vectorized operations instead of loops where possible.
    
    Parameters:
    df (pd.DataFrame): DataFrame to optimize operations for
    
    Returns:
    pd.DataFrame: DataFrame with optimized operations applied
    """
    # Example: Compute growth rate for all columns at once
    if 'daily_cases' in df.columns:
        df['growth_rate'] = df['daily_cases'].pct_change() * 100
    
    # Example: Use vectorized functions
    if 'date' in df.columns:
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
    
    return df

def load_and_cache_data(file_path, cache_path=None, force_reload=False):
    """
    Load data from file and cache it for faster access in subsequent runs.
    
    Parameters:
    file_path (str): Path to the data file
    cache_path (str): Path to save the cached data
    force_reload (bool): Whether to force reload from file even if cache exists
    
    Returns:
    pd.DataFrame: Loaded data
    """
    if cache_path is None:
        cache_path = f"{os.path.splitext(file_path)[0]}_cache.joblib"
    
    if os.path.exists(cache_path) and not force_reload:
        logger.info(f"Loading data from cache: {cache_path}")
        start_time = time.time()
        data = joblib.load(cache_path)
        logger.info(f"Data loaded from cache in {time.time() - start_time:.2f} seconds")
        return data
    
    logger.info(f"Loading data from file: {file_path}")
    start_time = time.time()
    
    # Determine file extension and load accordingly
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    elif file_path.endswith('.h5'):
        data = pd.read_hdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Optimize and cache data
    data = optimize_dataframe(data)
    
    logger.info(f"Caching optimized data to: {cache_path}")
    joblib.dump(data, cache_path)
    
    return data

def convert_to_efficient_format(input_path, output_path, format='parquet'):
    """
    Convert data to a more efficient storage format.
    
    Parameters:
    input_path (str): Path to input data file
    output_path (str): Path to save the converted data
    format (str): Output format ('parquet', 'hdf5', 'feather')
    
    Returns:
    bool: True if conversion was successful
    """
    logger.info(f"Converting {input_path} to {format} format")
    
    # Read input data
    if input_path.endswith('.csv'):
        data = pd.read_csv(input_path)
    elif input_path.endswith('.parquet'):
        data = pd.read_parquet(input_path)
    elif input_path.endswith('.h5'):
        data = pd.read_hdf(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    # Convert date columns
    date_columns = [col for col in data.columns if 'date' in col.lower()]
    for col in date_columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_datetime(data[col])
    
    # Optimize dataframe
    data = optimize_dataframe(data)
    
    # Save in target format
    if format == 'parquet':
        data.to_parquet(output_path, index=False)
    elif format == 'hdf5':
        data.to_hdf(output_path, key='data', mode='w')
    elif format == 'feather':
        data.to_feather(output_path)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logger.info(f"Data converted and saved to: {output_path}")
    return True

def optimize_visualization_data(data, max_points=1000):
    """
    Optimize data for visualization by reducing the number of points.
    
    Parameters:
    data (pd.DataFrame): Input data
    max_points (int): Maximum number of points to include
    
    Returns:
    pd.DataFrame: Optimized data for visualization
    """
    if len(data) <= max_points:
        return data
    
    # If data has a date column, use date-based sampling
    if 'date' in data.columns:
        # Sort by date
        data = data.sort_values('date')
        
        # Calculate sampling interval
        sample_interval = max(1, len(data) // max_points)
        
        # Sample data
        return data.iloc[::sample_interval].reset_index(drop=True)
    
    # Otherwise, use random sampling
    return data.sample(max_points).reset_index(drop=True)

def main():
    """Main function to demonstrate performance optimizations."""
    # Example data file
    file_path = '../data/covid19_data.csv'
    
    # Optimize data loading
    data = load_and_cache_data(file_path)
    
    # Convert to more efficient format
    convert_to_efficient_format(file_path, '../data/covid19_data.parquet', format='parquet')
    
    # Define a data processing function
    def process_data(df):
        # Example processing: calculate growth rates
        df = df.copy()
        if 'daily_cases' in df.columns:
            df['growth_rate'] = df['daily_cases'].pct_change() * 100
        return df
    
    # Process data in parallel
    logger.info("Processing data in parallel")
    parallel_result = parallel_process_dataframe(data, process_data)
    
    # Process data with Dask
    logger.info("Processing data with Dask")
    dask_result = dask_process_dataframe(data, process_data)
    
    # Optimize data for visualization
    logger.info("Optimizing data for visualization")
    viz_data = optimize_visualization_data(data)
    
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Visualization data shape: {viz_data.shape}")
    
    return {
        'original_data': data,
        'parallel_result': parallel_result,
        'dask_result': dask_result,
        'viz_data': viz_data
    }

if __name__ == "__main__":
    main() 