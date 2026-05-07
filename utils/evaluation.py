"""
Evaluation Utilities
====================
Functions for computing point prediction metrics and uncertainty
calibration metrics used throughout all experiments.
"""

import numpy as np
import pandas as pd


# =============================================================================
# Point Prediction Metrics
# =============================================================================

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Excludes samples where y_true == 0 to avoid division by zero.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =============================================================================
# Uncertainty Calibration Metrics
# =============================================================================

def compute_picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of true values that fall within [lower, upper].

    Parameters
    ----------
    y_true : array-like
        True values
    lower : array-like
        Lower bounds of prediction intervals
    upper : array-like
        Upper bounds of prediction intervals

    Returns
    -------
    float
        Coverage probability (0 to 1)
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)


def compute_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    return np.mean(upper - lower)


def compute_nmpiw(
    lower: np.ndarray,
    upper: np.ndarray,
    y_range: float
) -> float:
    """Normalized Mean Prediction Interval Width.

    MPIW divided by range of target variable.
    """
    return compute_mpiw(lower, upper) / y_range


def compute_calibration_error(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90
) -> float:
    """Calibration error: absolute difference between empirical coverage and target.

    |PICP - target_coverage|
    """
    picp = compute_picp(y_true, lower, upper)
    return abs(picp - target_coverage)


def compute_cwc(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90,
    eta: float = 50.0
) -> float:
    """Coverage Width-based Criterion.

    Penalizes both wide intervals and under-coverage.
    CWC = NMPIW * (1 + gamma * exp(-eta * (PICP - target)))
    where gamma = 1 if PICP < target, else 0.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    y_range = y_true.max() - y_true.min()
    if y_range == 0:
        y_range = 1.0

    nmpiw = compute_nmpiw(lower, upper, y_range)
    picp = compute_picp(y_true, lower, upper)

    if picp < target_coverage:
        penalty = np.exp(-eta * (picp - target_coverage))
    else:
        penalty = 0.0

    return nmpiw * (1 + penalty)


def compute_winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10
) -> float:
    """Winkler interval score.

    Rewards narrow intervals, penalizes misses.
    For each sample:
    - If y in [lower, upper]: score = upper - lower
    - If y < lower: score = (upper - lower) + (2/alpha) * (lower - y)
    - If y > upper: score = (upper - lower) + (2/alpha) * (y - upper)
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty = np.zeros_like(y_true, dtype=float)

    below = y_true < lower
    above = y_true > upper

    penalty[below] = (2.0 / alpha) * (lower[below] - y_true[below])
    penalty[above] = (2.0 / alpha) * (y_true[above] - upper[above])

    return np.mean(width + penalty)


# =============================================================================
# Combined Metrics
# =============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90
) -> dict:
    """Compute all point prediction and uncertainty metrics.

    Returns
    -------
    dict
        Dictionary with keys: MAE, RMSE, MAPE, PICP, MPIW, NMPIW,
        Calibration_Error, CWC, Winkler_Score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    y_range = y_true.max() - y_true.min()
    if y_range == 0:
        y_range = 1.0

    alpha = 1.0 - target_coverage

    return {
        'MAE': compute_mae(y_true, y_pred),
        'RMSE': compute_rmse(y_true, y_pred),
        'MAPE': compute_mape(y_true, y_pred),
        'PICP': compute_picp(y_true, lower, upper),
        'MPIW': compute_mpiw(lower, upper),
        'NMPIW': compute_nmpiw(lower, upper, y_range),
        'Calibration_Error': compute_calibration_error(y_true, lower, upper, target_coverage),
        'CWC': compute_cwc(y_true, lower, upper, target_coverage),
        'Winkler_Score': compute_winkler_score(y_true, lower, upper, alpha),
    }


def compute_metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    target_coverage: float = 0.90
) -> pd.DataFrame:
    """Compute all metrics grouped by a categorical variable.

    Parameters
    ----------
    groups : array-like
        Group labels (e.g., time period, route, day of week)

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group, columns for each metric
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    groups = np.asarray(groups)

    results = []
    for group in np.unique(groups):
        mask = groups == group
        n = mask.sum()

        metrics = compute_all_metrics(
            y_true[mask], y_pred[mask],
            lower[mask], upper[mask],
            target_coverage
        )
        metrics['group'] = group
        metrics['n_samples'] = n
        results.append(metrics)

    return pd.DataFrame(results).set_index('group')


def compute_rolling_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: np.ndarray,
    window_size: int = 100
) -> pd.DataFrame:
    """Compute rolling PICP over a sliding window of samples.

    Parameters
    ----------
    dates : array-like
        Dates for each sample (used for sorting)
    window_size : int
        Number of samples in the rolling window

    Returns
    -------
    pd.DataFrame
        DataFrame with date, rolling_picp, rolling_mpiw columns
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    dates = np.asarray(dates)

    # Sort by date
    sort_idx = np.argsort(dates)
    y_true = y_true[sort_idx]
    lower = lower[sort_idx]
    upper = upper[sort_idx]
    dates = dates[sort_idx]

    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    widths = upper - lower

    # Rolling computation
    rolling_picp = pd.Series(covered).rolling(window=window_size, min_periods=1).mean()
    rolling_mpiw = pd.Series(widths).rolling(window=window_size, min_periods=1).mean()

    return pd.DataFrame({
        'date': dates,
        'rolling_picp': rolling_picp.values,
        'rolling_mpiw': rolling_mpiw.values,
    })


def compute_daily_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: np.ndarray,
    target_coverage: float = 0.90
) -> pd.DataFrame:
    """Compute metrics aggregated by day.

    Returns
    -------
    pd.DataFrame
        One row per day with daily MAE, PICP, MPIW, etc.
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'lower': lower,
        'upper': upper,
        'date': dates,
    })

    results = []
    for date, group in df.groupby('date'):
        yt = group['y_true'].values
        yp = group['y_pred'].values
        lo = group['lower'].values
        up = group['upper'].values

        metrics = compute_all_metrics(yt, yp, lo, up, target_coverage)
        metrics['date'] = date
        metrics['n_samples'] = len(group)
        results.append(metrics)

    return pd.DataFrame(results).sort_values('date').reset_index(drop=True)
