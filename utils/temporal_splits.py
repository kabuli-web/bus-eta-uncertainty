"""
Temporal Split Utilities
========================
Functions for creating temporal train/calibration/test splits that
simulate distribution shift across different time periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Week boundary constants for the Astana bus dataset (Jul 29 - Sep 21, 2024)
WEEK_BOUNDARIES = {
    'W1': ('2024-07-29', '2024-08-04'),
    'W2': ('2024-08-05', '2024-08-11'),
    'W3': ('2024-08-12', '2024-08-18'),
    'W4': ('2024-08-19', '2024-08-25'),
    'W5': ('2024-08-26', '2024-09-01'),
    'W6': ('2024-09-02', '2024-09-08'),  # includes anomalous Sep 3-4
    'W7': ('2024-09-09', '2024-09-15'),
    'W8': ('2024-09-16', '2024-09-21'),  # partial week (6 days)
}

# Anomalous dates to exclude
ANOMALOUS_DATES = [pd.Timestamp('2024-09-03'), pd.Timestamp('2024-09-04')]

# Default temporal split assignments
DEFAULT_SPLIT = {
    'train': ['W1', 'W2', 'W3'],
    'calibration': ['W4'],
    'test_near': ['W5'],
    'test_mid': ['W6'],
    'test_far': ['W7', 'W8'],
}


def _get_date_range_for_weeks(weeks: list) -> tuple:
    """Get the overall date range for a list of weeks."""
    start_dates = [pd.Timestamp(WEEK_BOUNDARIES[w][0]) for w in weeks]
    end_dates = [pd.Timestamp(WEEK_BOUNDARIES[w][1]) for w in weeks]
    return min(start_dates), max(end_dates)


def _filter_by_weeks(df: pd.DataFrame, weeks: list) -> pd.DataFrame:
    """Filter DataFrame to only include rows within specified weeks."""
    masks = []
    for week in weeks:
        start, end = WEEK_BOUNDARIES[week]
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        mask = (df['date'] >= start) & (df['date'] <= end)
        masks.append(mask)

    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = combined_mask | m

    return df[combined_mask].reset_index(drop=True)


def get_temporal_split_static(
    df: pd.DataFrame,
    train_weeks: list = None,
    cal_weeks: list = None,
    test_weeks: list = None,
    exclude_anomalous: bool = True
) -> tuple:
    """Create a static temporal train/calibration/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'date' column
    train_weeks : list
        Week labels for training (default: W1-W3)
    cal_weeks : list
        Week labels for calibration (default: W4)
    test_weeks : list
        Week labels for testing (default: W5-W8)
    exclude_anomalous : bool
        Whether to exclude anomalous dates (Sep 3-4)

    Returns
    -------
    tuple
        (train_df, cal_df, test_df)
    """
    if train_weeks is None:
        train_weeks = DEFAULT_SPLIT['train']
    if cal_weeks is None:
        cal_weeks = DEFAULT_SPLIT['calibration']
    if test_weeks is None:
        test_weeks = DEFAULT_SPLIT['test_near'] + DEFAULT_SPLIT['test_mid'] + DEFAULT_SPLIT['test_far']

    # Exclude anomalous dates first
    if exclude_anomalous:
        df = df[~df['date'].isin(ANOMALOUS_DATES)].copy()

    train_df = _filter_by_weeks(df, train_weeks)
    cal_df = _filter_by_weeks(df, cal_weeks)
    test_df = _filter_by_weeks(df, test_weeks)

    print(f"Temporal split: Train={len(train_df):,} | "
          f"Cal={len(cal_df):,} | Test={len(test_df):,}")

    return train_df, cal_df, test_df


def get_temporal_split_by_period(
    df: pd.DataFrame,
    exclude_anomalous: bool = True
) -> dict:
    """Split data into all temporal periods.

    Returns
    -------
    dict
        Keys: 'train', 'calibration', 'test_near', 'test_mid', 'test_far'
        Values: DataFrames for each period
    """
    if exclude_anomalous:
        df = df[~df['date'].isin(ANOMALOUS_DATES)].copy()

    splits = {}
    for period, weeks in DEFAULT_SPLIT.items():
        splits[period] = _filter_by_weeks(df, weeks)
        print(f"  {period}: {len(splits[period]):,} records "
              f"({splits[period]['date'].nunique()} days)")

    return splits


def get_temporal_split_expanding_window(
    df: pd.DataFrame,
    initial_train_weeks: list = None,
    exclude_anomalous: bool = True
) -> list:
    """Create expanding window splits for temporal cross-validation.

    Each step adds one more week to training, uses next week as calibration,
    and subsequent weeks as test.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'date' column
    initial_train_weeks : list
        Starting training weeks (default: ['W1', 'W2'])
    exclude_anomalous : bool
        Whether to exclude anomalous dates

    Returns
    -------
    list
        List of (train_df, cal_df, test_df) tuples
    """
    if initial_train_weeks is None:
        initial_train_weeks = ['W1', 'W2']

    if exclude_anomalous:
        df = df[~df['date'].isin(ANOMALOUS_DATES)].copy()

    all_weeks = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
    splits = []

    start_idx = len(initial_train_weeks)
    for i in range(start_idx, len(all_weeks) - 1):
        train_weeks = all_weeks[:i]
        cal_week = [all_weeks[i]]
        test_weeks = all_weeks[i+1:]

        if not test_weeks:
            break

        train_df = _filter_by_weeks(df, train_weeks)
        cal_df = _filter_by_weeks(df, cal_week)
        test_df = _filter_by_weeks(df, test_weeks)

        splits.append((train_df, cal_df, test_df))

    return splits


def get_sliding_window_splits(
    df: pd.DataFrame,
    window_size_days: int = 7,
    step_days: int = 1,
    exclude_anomalous: bool = True
) -> list:
    """Generate daily (calibration_window, test_day) pairs for online CP.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'date' column
    window_size_days : int
        Number of days in the calibration window
    step_days : int
        Number of days to step forward each iteration
    exclude_anomalous : bool
        Whether to exclude anomalous dates

    Returns
    -------
    list
        List of (cal_window_df, test_day_df, test_date) tuples
    """
    if exclude_anomalous:
        df = df[~df['date'].isin(ANOMALOUS_DATES)].copy()

    dates = sorted(df['date'].unique())
    splits = []

    for i in range(window_size_days, len(dates), step_days):
        test_date = dates[i]
        cal_start = dates[max(0, i - window_size_days)]
        cal_end = dates[i - 1]

        cal_mask = (df['date'] >= cal_start) & (df['date'] <= cal_end)
        test_mask = df['date'] == test_date

        cal_df = df[cal_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        if len(cal_df) > 0 and len(test_df) > 0:
            splits.append((cal_df, test_df, test_date))

    return splits


def get_temporal_distance(
    test_df: pd.DataFrame,
    cal_end_date: str = '2024-08-25'
) -> pd.Series:
    """Compute days between each test sample and the calibration period end.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with 'date' column
    cal_end_date : str
        End date of the calibration period

    Returns
    -------
    pd.Series
        Number of days from calibration end for each sample
    """
    cal_end = pd.Timestamp(cal_end_date)
    return (test_df['date'] - cal_end).dt.days


def label_temporal_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'temporal_period' column based on date ranges.

    Labels: 'train', 'calibration', 'test_near', 'test_mid', 'test_far', 'excluded'

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'date' column

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'temporal_period' column
    """
    df = df.copy()
    df['temporal_period'] = 'unknown'

    for period, weeks in DEFAULT_SPLIT.items():
        for week in weeks:
            start = pd.Timestamp(WEEK_BOUNDARIES[week][0])
            end = pd.Timestamp(WEEK_BOUNDARIES[week][1])
            mask = (df['date'] >= start) & (df['date'] <= end)
            df.loc[mask, 'temporal_period'] = period

    # Mark anomalous dates
    df.loc[df['date'].isin(ANOMALOUS_DATES), 'temporal_period'] = 'excluded'

    return df


def get_week_label(date: pd.Timestamp) -> str:
    """Get the week label (W1-W8) for a given date.

    Parameters
    ----------
    date : pd.Timestamp
        Date to classify

    Returns
    -------
    str
        Week label or 'unknown'
    """
    for week, (start, end) in WEEK_BOUNDARIES.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return week
    return 'unknown'
