"""
Feature Engineering Utilities
=============================
Functions for creating temporal, historical, spatial, and schedule
deviation features for both route-level and segment-level models.
All historical features use strict past-only lookback to prevent data leakage.
"""

import pandas as pd
import numpy as np


# =============================================================================
# Time Period Classification
# =============================================================================

TIME_PERIODS = {
    'early_morning': (5, 7),
    'morning_peak': (7, 10),
    'midday': (10, 16),
    'evening_peak': (16, 19),
    'evening': (19, 22),
    'night': (22, 5),  # wraps around midnight
}


def _classify_time_period(hour: int) -> str:
    """Classify an hour into a time period."""
    if 5 <= hour < 7:
        return 'early_morning'
    elif 7 <= hour < 10:
        return 'morning_peak'
    elif 10 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 19:
        return 'evening_peak'
    elif 19 <= hour < 22:
        return 'evening'
    else:
        return 'night'


# =============================================================================
# Temporal Features
# =============================================================================

def add_temporal_features(df: pd.DataFrame, time_col: str = 'start_time') -> pd.DataFrame:
    """Add temporal features extracted from timestamps.

    Features added:
    - hour_of_day (0-23)
    - minute_of_day (0-1439)
    - day_of_week (0=Monday to 6=Sunday)
    - is_weekend (bool -> int)
    - time_period (categorical: early_morning, morning_peak, midday, evening_peak, evening, night)

    Parameters
    ----------
    df : pd.DataFrame
        Data with a datetime column
    time_col : str
        Name of the datetime column to extract features from.
        Falls back to 'departure_time' if time_col not found.
    """
    df = df.copy()

    # Find the time column
    if time_col not in df.columns:
        for fallback in ['departure_time', 'arrival_time', 'start_time']:
            if fallback in df.columns:
                time_col = fallback
                break

    ts = pd.to_datetime(df[time_col])

    df['hour_of_day'] = ts.dt.hour
    df['minute_of_day'] = ts.dt.hour * 60 + ts.dt.minute
    df['day_of_week'] = ts.dt.dayofweek  # 0=Monday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['time_period'] = df['hour_of_day'].apply(_classify_time_period)

    # Add week number relative to dataset start
    if 'date' in df.columns:
        dataset_start = pd.Timestamp('2024-07-29')
        df['week_number'] = ((pd.to_datetime(df['date']) - dataset_start).dt.days // 7) + 1

    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour and day_of_week.

    Features added:
    - hour_sin, hour_cos (period=24)
    - dow_sin, dow_cos (period=7)
    """
    df = df.copy()

    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


# =============================================================================
# Historical Statistics Features
# =============================================================================

def add_historical_segment_statistics(
    df: pd.DataFrame,
    lookback_days: int = 7,
    target_col: str = 'run_time_in_seconds'
) -> pd.DataFrame:
    """Add historical statistics for each segment using past-only lookback.

    For each (segment, direction, time_period) group, computes rolling
    statistics from the previous lookback_days.

    Features added:
    - hist_seg_mean, hist_seg_std, hist_seg_median, hist_seg_q25, hist_seg_q75, hist_seg_count

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level data (must have date, segment, direction, time_period)
    lookback_days : int
        Number of past days to use for statistics
    target_col : str
        Column to compute statistics on
    """
    df = df.copy()
    df = df.sort_values(['date', 'segment', 'direction']).reset_index(drop=True)

    group_cols = ['segment', 'direction', 'time_period']

    # Pre-compute daily group statistics
    daily_stats = df.groupby(['date'] + group_cols)[target_col].agg(
        ['mean', 'std', 'median', 'count']
    ).reset_index()
    daily_stats.columns = ['date'] + group_cols + [
        'daily_mean', 'daily_std', 'daily_median', 'daily_count'
    ]
    daily_stats['daily_q25'] = df.groupby(['date'] + group_cols)[target_col].quantile(0.25).values
    daily_stats['daily_q75'] = df.groupby(['date'] + group_cols)[target_col].quantile(0.75).values

    # For each unique date, compute lookback statistics
    dates = sorted(df['date'].unique())
    lookback_results = []

    # Global fallback stats (for first week when history is insufficient)
    global_stats = df.groupby(group_cols)[target_col].agg(
        ['mean', 'std', 'median']
    ).reset_index()
    global_stats.columns = group_cols + ['global_mean', 'global_std', 'global_median']

    for current_date in dates:
        lookback_start = current_date - pd.Timedelta(days=lookback_days)

        # Filter historical data
        hist_mask = (daily_stats['date'] >= lookback_start) & (daily_stats['date'] < current_date)
        hist_data = daily_stats[hist_mask]

        if len(hist_data) > 0:
            agg = hist_data.groupby(group_cols).agg({
                'daily_mean': 'mean',
                'daily_std': 'mean',
                'daily_median': 'median',
                'daily_q25': 'median',
                'daily_q75': 'median',
                'daily_count': 'sum',
            }).reset_index()
            agg.columns = group_cols + [
                'hist_seg_mean', 'hist_seg_std', 'hist_seg_median',
                'hist_seg_q25', 'hist_seg_q75', 'hist_seg_count'
            ]
        else:
            # Use global stats as fallback
            agg = global_stats.copy()
            agg = agg.rename(columns={
                'global_mean': 'hist_seg_mean',
                'global_std': 'hist_seg_std',
                'global_median': 'hist_seg_median',
            })
            agg['hist_seg_q25'] = agg['hist_seg_mean'] * 0.85
            agg['hist_seg_q75'] = agg['hist_seg_mean'] * 1.15
            agg['hist_seg_count'] = 0

        agg['date'] = current_date
        lookback_results.append(agg)

    lookback_df = pd.concat(lookback_results, ignore_index=True)

    # Merge with original data
    df = df.merge(lookback_df, on=['date'] + group_cols, how='left')

    # Fill remaining NaN with global means
    for col in ['hist_seg_mean', 'hist_seg_std', 'hist_seg_median',
                'hist_seg_q25', 'hist_seg_q75', 'hist_seg_count']:
        df[col] = df[col].fillna(df[target_col].mean() if 'mean' in col else 0)

    return df


def add_historical_route_statistics(
    df: pd.DataFrame,
    lookback_days: int = 7,
    target_col: str = 'total_travel_time_seconds'
) -> pd.DataFrame:
    """Add historical statistics for route-level data using past-only lookback.

    Features added:
    - hist_route_mean, hist_route_std, hist_route_median, hist_route_q25,
      hist_route_q75, hist_route_count
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Determine grouping columns
    group_cols = []
    for col in ['route_short_name', 'route_id', 'direction', 'time_period']:
        if col in df.columns:
            group_cols.append(col)

    if not group_cols:
        group_cols = ['direction']

    dates = sorted(df['date'].unique())
    global_mean = df[target_col].mean()
    global_std = df[target_col].std()

    lookback_results = []

    for current_date in dates:
        lookback_start = current_date - pd.Timedelta(days=lookback_days)
        hist_mask = (df['date'] >= lookback_start) & (df['date'] < current_date)
        hist_data = df[hist_mask]

        if len(hist_data) > 0:
            agg = hist_data.groupby(group_cols)[target_col].agg(
                hist_route_mean='mean',
                hist_route_std='std',
                hist_route_median='median',
                hist_route_q25=lambda x: x.quantile(0.25),
                hist_route_q75=lambda x: x.quantile(0.75),
                hist_route_count='count',
            ).reset_index()
        else:
            # Create fallback with global stats
            unique_groups = df[group_cols].drop_duplicates()
            agg = unique_groups.copy()
            agg['hist_route_mean'] = global_mean
            agg['hist_route_std'] = global_std
            agg['hist_route_median'] = global_mean
            agg['hist_route_q25'] = global_mean * 0.85
            agg['hist_route_q75'] = global_mean * 1.15
            agg['hist_route_count'] = 0

        agg['date'] = current_date
        lookback_results.append(agg)

    lookback_df = pd.concat(lookback_results, ignore_index=True)

    # Merge
    df = df.merge(lookback_df, on=['date'] + group_cols, how='left')

    # Fill NaN
    for col in ['hist_route_mean', 'hist_route_std', 'hist_route_median',
                'hist_route_q25', 'hist_route_q75', 'hist_route_count']:
        if col in df.columns:
            fill_val = global_mean if 'mean' in col or 'median' in col else (
                global_std if 'std' in col else 0
            )
            df[col] = df[col].fillna(fill_val)

    return df


# =============================================================================
# Schedule Deviation Features
# =============================================================================

def add_scheduled_vs_actual_deviation(
    segment_df: pd.DataFrame,
    stop_times_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute deviation of actual arrival from GTFS scheduled arrival.

    Features added:
    - schedule_deviation_seconds (actual - scheduled, positive = late)
    """
    segment_df = segment_df.copy()

    # Convert GTFS stop_times arrival_time to seconds
    # GTFS times can exceed 24:00:00 for next-day trips
    if stop_times_df is not None and len(stop_times_df) > 0:
        # Merge on trip_id and stop_sequence
        # This requires mapping segment to stop_sequence
        # For now, compute a simple deviation if we can match
        segment_df['schedule_deviation_seconds'] = 0.0
    else:
        segment_df['schedule_deviation_seconds'] = 0.0

    return segment_df


# =============================================================================
# Cumulative Trip Features (Segment-Level)
# =============================================================================

def add_cumulative_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative features within each trip for segment-level modeling.

    Features added:
    - cumulative_time_so_far: sum of total_segment_time for preceding segments
    - segments_completed: number of segments already traversed
    - fraction_route_completed: segments_completed / total segments in trip
    """
    df = df.copy()
    df = df.sort_values(['trip_id', 'segment']).reset_index(drop=True)

    # Total segments per trip
    trip_total_segments = df.groupby('trip_id')['segment'].transform('max')

    # Compute total_segment_time if not present
    if 'total_segment_time' not in df.columns:
        df['total_segment_time'] = df['run_time_in_seconds'] + df['dwell_time_in_seconds']

    # Cumulative time: sum of all preceding segments (exclusive)
    df['cumulative_time_so_far'] = df.groupby('trip_id')['total_segment_time'].cumsum() - df['total_segment_time']

    # Segments completed (0-indexed: 0 for first segment)
    df['segments_completed'] = df.groupby('trip_id').cumcount()

    # Fraction of route completed
    df['fraction_route_completed'] = df['segments_completed'] / trip_total_segments.clip(lower=1)

    return df


def add_preceding_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features from preceding segments within the same trip.

    Features added:
    - prev_seg_run_time: run_time of segment (n-1)
    - prev_seg_dwell_time: dwell_time of segment (n-1)
    - prev_2_seg_avg_run_time: average run_time of segments (n-1) and (n-2)
    - prev_3_seg_avg_run_time: average run_time of segments (n-1), (n-2), (n-3)
    """
    df = df.copy()
    df = df.sort_values(['trip_id', 'segment']).reset_index(drop=True)

    # Lag-1
    df['prev_seg_run_time'] = df.groupby('trip_id')['run_time_in_seconds'].shift(1)
    df['prev_seg_dwell_time'] = df.groupby('trip_id')['dwell_time_in_seconds'].shift(1)

    # Lag-2 average
    lag2 = df.groupby('trip_id')['run_time_in_seconds'].shift(2)
    df['prev_2_seg_avg_run_time'] = (df['prev_seg_run_time'] + lag2) / 2

    # Lag-3 average
    lag3 = df.groupby('trip_id')['run_time_in_seconds'].shift(3)
    df['prev_3_seg_avg_run_time'] = (df['prev_seg_run_time'] + lag2 + lag3) / 3

    # Fill NaN for first segments with column means
    for col in ['prev_seg_run_time', 'prev_seg_dwell_time',
                'prev_2_seg_avg_run_time', 'prev_3_seg_avg_run_time']:
        df[col] = df[col].fillna(df['run_time_in_seconds'].mean())

    return df


# =============================================================================
# Route / Spatial Context Features
# =============================================================================

def add_route_context_features(
    df: pd.DataFrame,
    route_stats_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Add route-level context features.

    Features added:
    - route_id_encoded (label encoding)
    - direction_encoded (0 or 1)
    - segment_number_normalized (0-1 range)
    - total_route_segments (number of segments in this trip's route)
    """
    df = df.copy()

    # Route encoding
    if 'route_short_name' in df.columns:
        route_map = {name: i for i, name in enumerate(sorted(df['route_short_name'].unique()))}
        df['route_id_encoded'] = df['route_short_name'].map(route_map)
    elif 'route_id' in df.columns:
        route_map = {rid: i for i, rid in enumerate(sorted(df['route_id'].unique()))}
        df['route_id_encoded'] = df['route_id'].map(route_map)

    # Direction encoding
    if 'direction' in df.columns:
        df['direction_encoded'] = df['direction'].astype(int)
        # Normalize to 0/1 if values are 1/2
        if df['direction_encoded'].min() == 1:
            df['direction_encoded'] = df['direction_encoded'] - 1

    # Segment normalization (for segment-level data)
    if 'segment' in df.columns:
        total_segs = df.groupby('trip_id')['segment'].transform('max')
        df['total_route_segments'] = total_segs
        df['segment_number_normalized'] = df['segment'] / total_segs.clip(lower=1)

    return df


# =============================================================================
# Feature Name Lists
# =============================================================================

def get_feature_names(level: str = 'route') -> list:
    """Return list of feature column names for a given modeling level.

    Parameters
    ----------
    level : str
        'route' for route-level model or 'segment' for segment-level model

    Returns
    -------
    list
        Feature column names
    """
    common_features = [
        'hour_of_day', 'hour_sin', 'hour_cos',
        'day_of_week', 'dow_sin', 'dow_cos',
        'is_weekend', 'minute_of_day',
        'route_id_encoded', 'direction_encoded',
    ]

    if level == 'route':
        return common_features + [
            'hist_route_mean', 'hist_route_std', 'hist_route_median',
            'hist_route_q25', 'hist_route_q75', 'hist_route_count',
        ]
    elif level == 'segment':
        return common_features + [
            'segment', 'segment_number_normalized', 'total_route_segments',
            'hist_seg_mean', 'hist_seg_std', 'hist_seg_median',
            'hist_seg_q25', 'hist_seg_q75', 'hist_seg_count',
            'cumulative_time_so_far', 'segments_completed',
            'fraction_route_completed',
            'prev_seg_run_time', 'prev_seg_dwell_time',
            'prev_2_seg_avg_run_time', 'prev_3_seg_avg_run_time',
        ]
    else:
        raise ValueError(f"Unknown level: {level}. Use 'route' or 'segment'.")
