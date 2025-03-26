import pandas as pd
import numpy as np

def create_date_features(df: pd.DataFrame, date_column='date'):
    """
    Create day, month_sin/cos, year, day_of_week_sin/cos, week_of_year features from the date feature. 
    Making sure that there is not data leakage

    Args: 
        df: pd.DataFrame
        date_column: str (by defoult 'date')

    Return: 
        df: pd.DataFrame (with the newly created custom feautres)
    """
    df[date_column] = pd.to_datetime(df[date_column])

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["week_of_year"] = df[date_column].dt.isocalendar().week 

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df

def create_lag_features(df: pd.DataFrame, groupe_columns, target_column, lags=[1,7,14]):
    """
    Creates lag features for a given target column, grouped by specific columns.

    Args:
        df (pd.DataFrame): Input dataframe containing the time series data.
        groupe_columns (list): Columns to group by before creating lag features (e.g., ['store', 'item']).
        target_column (str): The name of the target variable to generate lags for.
        lags (list): List of integers representing the lag intervals to compute.

    Returns:
        pd.DataFrame: DataFrame with new lag features added (e.g., 'sales_lag_1', 'sales_lag_7', etc.).
    """
    for lag in lags: 
        df[f"{target_column}_lag_{lag}"] = df.groupby(by=groupe_columns)[target_column].shift(lag)

    return df

def create_rolling_features(df, group_columns, target_column, windows=[7,14,30]):
    """
    Creates rolling mean features for a given target column, based on groupings and window sizes.

    Args:
        df (pd.DataFrame): Input dataframe with time series data.
        group_columns (list): Columns to group by before computing rolling statistics (e.g., ['store', 'item']).
        target_column (str): The name of the target variable to compute rolling means for.
        windows (list): List of window sizes (in time steps) to use for calculating rolling means.

    Returns:
        pd.DataFrame: DataFrame with new rolling mean features added (e.g., 'sales_rolling_7', etc.).
    """
    for window in windows: 
        df[f"{target_column}_rolling_{window}"] = (df.groupby(by=group_columns)[target_column].shift(1).rolling(window=window).mean().reset_index(level=0, drop=True))
    return df

def create_all_features(df):
    """
    Creates and combines all relevant time series features: 
    date-based, lag-based, and rolling average features.

    The function assumes the DataFrame contains 'store', 'item', 'sales', and 'date' columns.

    Args:
        df (pd.DataFrame): Input dataframe with raw data.

    Returns:
        pd.DataFrame: DataFrame with enriched features for time series modeling.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store", "item", "date"])

    df = create_date_features(df, date_column='date')
    df = create_lag_features(df, group_columns=['store', 'item'], target_column='sales', lags=[1, 7, 14])
    df = create_rolling_features(df, group_columns=['store', 'item'], target_column='sales', windows=[7, 14, 30])
    return df 