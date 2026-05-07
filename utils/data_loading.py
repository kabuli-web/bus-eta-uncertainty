"""
Data Loading Utilities
======================
Functions for loading segment-level data, GTFS files, joining datasets,
and caching processed DataFrames as parquet files.
"""

import os
import pandas as pd
import numpy as np


def load_segment_data(filepath: str) -> pd.DataFrame:
    """Load segment_level_data.csv with proper dtypes and parsed datetime columns.

    Parameters
    ----------
    filepath : str
        Path to segment_level_data.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates and times
    """
    df = pd.read_csv(
        filepath,
        parse_dates=['date', 'start_time', 'arrival_time', 'departure_time'],
        dtype={
            'deviceid': 'int32',
            'direction': 'int8',
            'segment': 'int16',
            'start_point': str,
            'end_point': str,
            'run_time_in_seconds': 'float64',
            'dwell_time_in_seconds': 'float64',
            'trip_id': 'int64',
            'device_guid': str,
            'start_guid': str,
            'end_guid': str,
        }
    )
    # Ensure date is date-only (no time component)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df


def load_gtfs_trips(gtfs_dir: str) -> pd.DataFrame:
    """Load GTFS trips.txt file.

    Returns DataFrame with: trip_id, route_id, service_id, direction_id,
    start_time, end_time, vehicle_id
    """
    filepath = os.path.join(gtfs_dir, 'trips.txt')
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
    return df


def load_gtfs_stops(gtfs_dir: str) -> pd.DataFrame:
    """Load GTFS stops.txt file.

    Returns DataFrame with: stop_id, stop_name, stop_lat, stop_lon
    """
    filepath = os.path.join(gtfs_dir, 'stops.txt')
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
    return df


def load_gtfs_stop_times(gtfs_dir: str) -> pd.DataFrame:
    """Load GTFS stop_times.txt file.

    Returns DataFrame with: trip_id, arrival_time, departure_time,
    stop_id, stop_sequence
    """
    filepath = os.path.join(gtfs_dir, 'stop_times.txt')
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
    return df


def load_gtfs_routes(gtfs_dir: str) -> pd.DataFrame:
    """Load GTFS routes.txt file.

    Returns DataFrame with: route_id, agency_id, route_long_name,
    route_type, route_short_name
    """
    filepath = os.path.join(gtfs_dir, 'routes.txt')
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
    return df


def load_gtfs_calendar_dates(gtfs_dir: str) -> pd.DataFrame:
    """Load GTFS calendar_dates.txt file.

    Returns DataFrame with: service_id, date, exception_type
    """
    filepath = os.path.join(gtfs_dir, 'calendar_dates.txt')
    df = pd.read_csv(filepath, sep='\t', encoding='latin-1', parse_dates=['date'])
    return df


def join_segment_with_gtfs(
    segment_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> pd.DataFrame:
    """Join segment data with GTFS trips and routes to add route information.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Segment-level data with trip_id column
    trips_df : pd.DataFrame
        GTFS trips with trip_id and route_id
    routes_df : pd.DataFrame
        GTFS routes with route_id and route_short_name

    Returns
    -------
    pd.DataFrame
        Segment data with added route_id and route_short_name columns
    """
    # trips_df has trip_id as integer in segment data but may need mapping
    # First, create a trip lookup from trips_df
    trip_lookup = trips_df[['trip_id', 'route_id']].copy()

    # Join segment data with trip info
    merged = segment_df.merge(trip_lookup, on='trip_id', how='left')

    # Join with routes to get route_short_name
    route_lookup = routes_df[['route_id', 'route_short_name', 'route_long_name']].copy()
    merged = merged.merge(route_lookup, on='route_id', how='left')

    return merged


def cache_dataframe(df: pd.DataFrame, cache_path: str) -> None:
    """Save DataFrame to parquet for fast reload.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to cache
    cache_path : str
        Path to save the parquet file
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False, engine='pyarrow')


def load_cached_dataframe(cache_path: str) -> pd.DataFrame:
    """Load DataFrame from parquet cache.

    Parameters
    ----------
    cache_path : str
        Path to the parquet file

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return pd.read_parquet(cache_path, engine='pyarrow')
