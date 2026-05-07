"""
Preprocessing Utilities
=======================
Functions for data cleaning, outlier detection/removal, trip filtering,
and route-level aggregation.
"""

import pandas as pd
import numpy as np


def remove_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_removed = n_before - len(df_clean)
    print(f"Removed {n_removed:,} duplicate records ({n_removed/n_before*100:.2f}%)")
    return df_clean.reset_index(drop=True)


def detect_and_remove_outliers(
    df: pd.DataFrame,
    column: str = 'run_time_in_seconds',
    method: str = 'iqr',
    threshold: float = 1.5,
    group_cols: list = None
) -> tuple:
    """Detect and remove outliers from a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column to check for outliers
    method : str
        'iqr' for IQR-based or 'zscore' for z-score based detection
    threshold : float
        For IQR: multiplier (default 1.5). For zscore: number of std devs (default 3.0)
    group_cols : list, optional
        Columns to group by before computing outlier bounds (e.g., ['segment', 'direction'])

    Returns
    -------
    tuple
        (cleaned_df, outliers_df) - cleaned data and removed outliers
    """
    if group_cols is None:
        group_cols = []

    if method == 'iqr':
        if group_cols:
            # Compute per-group bounds
            bounds = df.groupby(group_cols)[column].agg(
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75)
            ).reset_index()
            bounds['iqr'] = bounds['q3'] - bounds['q1']
            bounds['lower'] = bounds['q1'] - threshold * bounds['iqr']
            bounds['upper'] = bounds['q3'] + threshold * bounds['iqr']

            merged = df.merge(bounds[group_cols + ['lower', 'upper']], on=group_cols, how='left')
            mask = (merged[column] >= merged['lower']) & (merged[column] <= merged['upper'])

            cleaned_df = df[mask.values].reset_index(drop=True)
            outliers_df = df[~mask.values].reset_index(drop=True)
        else:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

            mask = (df[column] >= lower) & (df[column] <= upper)
            cleaned_df = df[mask].reset_index(drop=True)
            outliers_df = df[~mask].reset_index(drop=True)

    elif method == 'zscore':
        if group_cols:
            stats = df.groupby(group_cols)[column].agg(['mean', 'std']).reset_index()
            stats.columns = group_cols + ['group_mean', 'group_std']
            merged = df.merge(stats, on=group_cols, how='left')
            merged['zscore'] = (merged[column] - merged['group_mean']) / merged['group_std'].clip(lower=1e-10)
            mask = merged['zscore'].abs() <= threshold

            cleaned_df = df[mask.values].reset_index(drop=True)
            outliers_df = df[~mask.values].reset_index(drop=True)
        else:
            mean = df[column].mean()
            std = df[column].std()
            zscore = (df[column] - mean) / std
            mask = zscore.abs() <= threshold
            cleaned_df = df[mask].reset_index(drop=True)
            outliers_df = df[~mask].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

    n_removed = len(outliers_df)
    print(f"Outlier removal ({method}, threshold={threshold}): "
          f"removed {n_removed:,} records ({n_removed/len(df)*100:.2f}%)")

    return cleaned_df, outliers_df


def filter_incomplete_trips(
    df: pd.DataFrame,
    min_segments: int = 30
) -> pd.DataFrame:
    """Remove trips with fewer than min_segments segments.

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level data
    min_segments : int
        Minimum number of segments required for a complete trip

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    trip_counts = df.groupby('trip_id')['segment'].nunique()
    valid_trips = trip_counts[trip_counts >= min_segments].index

    n_before = df['trip_id'].nunique()
    df_filtered = df[df['trip_id'].isin(valid_trips)].reset_index(drop=True)
    n_after = df_filtered['trip_id'].nunique()

    print(f"Trip filtering (min_segments={min_segments}): "
          f"{n_before:,} -> {n_after:,} trips "
          f"(removed {n_before - n_after:,}, {(n_before - n_after)/n_before*100:.1f}%)")

    return df_filtered


def filter_anomalous_dates(
    df: pd.DataFrame,
    min_daily_records: int = 5000
) -> pd.DataFrame:
    """Remove dates with anomalously low record counts.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'date' column
    min_daily_records : int
        Minimum records per day to keep

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    daily_counts = df.groupby('date').size()
    anomalous_dates = daily_counts[daily_counts < min_daily_records].index.tolist()

    if anomalous_dates:
        date_strs = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
                     for d in anomalous_dates]
        print(f"Removing anomalous dates: {date_strs} "
              f"(< {min_daily_records:,} records each)")

    n_before = len(df)
    df_filtered = df[~df['date'].isin(anomalous_dates)].reset_index(drop=True)
    n_removed = n_before - len(df_filtered)
    print(f"Removed {n_removed:,} records from {len(anomalous_dates)} anomalous dates")

    return df_filtered


def compute_segment_travel_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total segment time as run_time + dwell_time.

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level data

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'total_segment_time' column
    """
    df = df.copy()
    df['run_time_in_seconds'] = pd.to_numeric(df['run_time_in_seconds'], errors='coerce')
    df['dwell_time_in_seconds'] = pd.to_numeric(df['dwell_time_in_seconds'], errors='coerce')
    df['total_segment_time'] = df['run_time_in_seconds'] + df['dwell_time_in_seconds']
    return df


def aggregate_to_route_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-level data to route-level (one row per trip).

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level data with at least: trip_id, direction, date,
        run_time_in_seconds, dwell_time_in_seconds, segment

    Returns
    -------
    pd.DataFrame
        Route-level DataFrame with columns: trip_id, route_id, route_short_name,
        direction, date, departure_time, total_travel_time_seconds,
        total_run_time_seconds, total_dwell_time_seconds, num_segments
    """
    # Determine which columns are available for groupby
    group_cols = ['trip_id']
    agg_dict = {
        'run_time_in_seconds': 'sum',
        'dwell_time_in_seconds': 'sum',
        'segment': 'nunique',
    }

    # Add optional columns if they exist
    first_cols = {}
    for col in ['direction', 'date', 'route_id', 'route_short_name', 'deviceid']:
        if col in df.columns:
            first_cols[col] = 'first'

    agg_dict.update(first_cols)

    # Get departure time as the minimum start_time per trip
    if 'start_time' in df.columns:
        agg_dict['start_time'] = 'min'

    route_df = df.groupby('trip_id').agg(agg_dict).reset_index()

    # Rename columns
    route_df = route_df.rename(columns={
        'run_time_in_seconds': 'total_run_time_seconds',
        'dwell_time_in_seconds': 'total_dwell_time_seconds',
        'segment': 'num_segments',
        'start_time': 'departure_time',
    })

    # Compute total travel time
    route_df['total_travel_time_seconds'] = (
        route_df['total_run_time_seconds'] + route_df['total_dwell_time_seconds']
    )

    return route_df


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame

    Returns
    -------
    dict
        Report with missing values, dtypes, value ranges, etc.
    """
    report = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    # Numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report[f'{col}_stats'] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'zeros': (df[col] == 0).sum(),
            'negatives': (df[col] < 0).sum(),
        }

    return report
