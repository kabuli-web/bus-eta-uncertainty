"""
Visualization Utilities
=======================
Publication-quality plotting functions for thesis figures.
All figures use a consistent style with serif fonts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import os


# =============================================================================
# Style Configuration
# =============================================================================

THESIS_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palette for methods comparison
METHOD_COLORS = {
    'Static CP': '#1f77b4',
    'Online Expanding': '#ff7f0e',
    'Online Sliding-7d': '#2ca02c',
    'Online Sliding-14d': '#d62728',
    # Hourly update variants (Phase 5b)
    'Hourly Expanding': '#e377c2',
    'Hourly Sliding-7d': '#17becf',
    'Hourly Sliding-14d': '#bcbd22',
    # Daily vs Hourly comparison aliases
    'Daily Expanding': '#ff7f0e',
    'Daily Sliding-7d': '#2ca02c',
    'Daily Sliding-14d': '#d62728',
}

# Color palette for temporal periods
PERIOD_COLORS = {
    'train': '#4e79a7',
    'calibration': '#59a14f',
    'test_near': '#f28e2b',
    'test_mid': '#e15759',
    'test_far': '#b07aa1',
    'excluded': '#bab0ac',
}


def set_thesis_style():
    """Apply matplotlib rcParams for thesis-quality figures."""
    plt.rcParams.update(THESIS_STYLE)
    sns.set_palette("colorblind")


def _save_figure(fig, save_path):
    """Save figure as both PDF and PNG if path provided."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        # Also save PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')


# =============================================================================
# Time Series and Interval Plots
# =============================================================================

def plot_time_series_with_intervals(
    dates, y_true, y_pred, lower, upper,
    title='Predictions with Confidence Intervals',
    ylabel='Travel Time (seconds)',
    save_path=None
):
    """Time series plot showing actual vs predicted with shaded prediction intervals."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    idx = np.arange(len(y_true))

    ax.fill_between(idx, lower, upper, alpha=0.25, color='#2ca02c', label='Prediction Interval')
    ax.plot(idx, y_true, 'k.', markersize=3, alpha=0.6, label='Actual')
    ax.plot(idx, y_pred, 'b-', linewidth=0.8, alpha=0.7, label='Predicted')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_coverage_over_time(
    daily_coverages, target_coverage=0.90,
    title='Daily Empirical Coverage (PICP)',
    save_path=None
):
    """Line plot of daily PICP with horizontal target line."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    if isinstance(daily_coverages, pd.DataFrame):
        dates = daily_coverages['date']
        picp = daily_coverages['PICP']
    else:
        dates = range(len(daily_coverages))
        picp = daily_coverages

    ax.plot(dates, picp, 'o-', markersize=4, linewidth=1.5, color='#1f77b4', label='Daily PICP')
    ax.axhline(y=target_coverage, color='red', linestyle='--', linewidth=1.5,
               label=f'Target ({target_coverage:.0%})')

    # Shade under-coverage region
    ax.fill_between(dates, 0, target_coverage, alpha=0.05, color='red')

    ax.set_xlabel('Date')
    ax.set_ylabel('PICP')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()

    if hasattr(dates.iloc[0] if isinstance(dates, pd.Series) else dates[0], 'strftime'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_interval_width_distribution(
    widths, title='Prediction Interval Width Distribution',
    xlabel='Interval Width (seconds)', save_path=None
):
    """Histogram + KDE of prediction interval widths."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(widths, bins=50, density=True, alpha=0.6, color='#1f77b4', edgecolor='white')

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(widths)
    x_range = np.linspace(min(widths), max(widths), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    ax.axvline(np.mean(widths), color='green', linestyle='--',
               label=f'Mean: {np.mean(widths):.1f}s')
    ax.axvline(np.median(widths), color='orange', linestyle=':',
               label=f'Median: {np.median(widths):.1f}s')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# =============================================================================
# Comparison and Bar Charts
# =============================================================================

def plot_calibration_comparison_bar(
    results_df, metric='PICP', group_col='temporal_period',
    title='Metric Comparison', target_line=None, save_path=None
):
    """Grouped bar chart comparing metrics across methods/periods."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    results_df.plot(kind='bar', y=metric, ax=ax, color=sns.color_palette("colorblind"))

    if target_line is not None:
        ax.axhline(y=target_line, color='red', linestyle='--', linewidth=1.5,
                   label=f'Target ({target_line})')

    ax.set_xlabel(group_col.replace('_', ' ').title())
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_coverage_vs_temporal_distance(
    temporal_distances, coverages,
    title='Coverage vs Temporal Distance from Calibration',
    target_coverage=0.90, save_path=None
):
    """Scatter/line plot showing how coverage degrades with temporal distance."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(temporal_distances, coverages, alpha=0.4, s=20, color='#1f77b4')

    # Add trend line (LOWESS or rolling mean)
    df_temp = pd.DataFrame({'dist': temporal_distances, 'cov': coverages})
    df_temp = df_temp.sort_values('dist')
    rolling = df_temp.groupby('dist')['cov'].mean()
    ax.plot(rolling.index, rolling.values, 'r-', linewidth=2, label='Daily Mean')

    ax.axhline(y=target_coverage, color='green', linestyle='--',
               label=f'Target ({target_coverage:.0%})')

    ax.set_xlabel('Days Since Calibration End')
    ax.set_ylabel('PICP')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# =============================================================================
# Segment-Level Plots
# =============================================================================

def plot_segment_uncertainty_heatmap(
    segment_uncertainties,
    title='Segment-Level Uncertainty Attribution',
    save_path=None
):
    """Heatmap showing uncertainty contribution of each segment along a route."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    if isinstance(segment_uncertainties, pd.DataFrame):
        data = segment_uncertainties.pivot_table(
            values='mean_width', index='direction', columns='segment'
        )
        sns.heatmap(data, ax=ax, cmap='YlOrRd', annot=False,
                    cbar_kws={'label': 'Mean Interval Width (s)'})

    ax.set_title(title)
    ax.set_xlabel('Segment Number')
    ax.set_ylabel('Direction')

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_segment_waterfall(
    segment_contributions,
    title='Cumulative Uncertainty Build-Up Along Route',
    save_path=None
):
    """Waterfall chart showing cumulative uncertainty build-up along route segments."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    segments = segment_contributions['segment']
    widths = segment_contributions['mean_width']
    cumulative = np.cumsum(widths)

    colors = plt.cm.YlOrRd(widths / widths.max())

    ax.bar(segments, widths, bottom=cumulative - widths, color=colors,
           edgecolor='white', linewidth=0.5)
    ax.plot(segments, cumulative, 'k--', linewidth=1.5, label='Cumulative')

    ax.set_xlabel('Segment Number')
    ax.set_ylabel('Interval Width (seconds)')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# =============================================================================
# Model Analysis Plots
# =============================================================================

def plot_feature_importance(
    feature_names, importances, top_n=20,
    title='Feature Importance', save_path=None
):
    """Horizontal bar chart of top-N feature importances."""
    set_thesis_style()

    # Sort by importance
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    ax.barh(range(len(idx)), importances[idx], color='#1f77b4')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_residual_analysis(
    y_true, y_pred, title='Residual Analysis', save_path=None
):
    """4-panel residual analysis: residual vs predicted, histogram, QQ, residual vs index."""
    set_thesis_style()

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title('Residuals vs Predicted')

    # Panel 2: Residual Histogram
    axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='white')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Residual Distribution')

    # Panel 3: QQ Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # Panel 4: Residuals vs Index (time order)
    axes[1, 1].plot(residuals, '.', markersize=2, alpha=0.3)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('Residuals Over Time')

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_data_distribution_comparison(
    train_values, test_values, feature_name='Travel Time',
    title=None, save_path=None
):
    """Overlapping histograms/KDEs of train vs test feature distributions."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    if title is None:
        title = f'Distribution Comparison: {feature_name}'

    ax.hist(train_values, bins=50, density=True, alpha=0.5,
            color='#4e79a7', label='Train', edgecolor='white')
    ax.hist(test_values, bins=50, density=True, alpha=0.5,
            color='#e15759', label='Test', edgecolor='white')

    # KDE overlay
    from scipy.stats import gaussian_kde
    x_min = min(min(train_values), min(test_values))
    x_max = max(max(train_values), max(test_values))
    x_range = np.linspace(x_min, x_max, 200)

    kde_train = gaussian_kde(train_values)
    kde_test = gaussian_kde(test_values)
    ax.plot(x_range, kde_train(x_range), '-', color='#4e79a7', linewidth=2)
    ax.plot(x_range, kde_test(x_range), '-', color='#e15759', linewidth=2)

    ax.set_xlabel(feature_name)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# =============================================================================
# Table Export
# =============================================================================

def create_summary_table(results, save_path=None):
    """Create formatted summary table, optionally save as LaTeX and CSV.

    Parameters
    ----------
    results : dict or pd.DataFrame
        Results data
    save_path : str, optional
        Base path (without extension) to save the table

    Returns
    -------
    pd.DataFrame
        Formatted results table
    """
    if isinstance(results, dict):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save CSV
        df.to_csv(save_path + '.csv', index=True)
        # Save LaTeX
        latex = df.to_latex(float_format='%.4f', caption='Results Summary')
        with open(save_path + '.tex', 'w') as f:
            f.write(latex)

    return df
