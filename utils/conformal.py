"""
Conformal Prediction Utilities
==============================
Wrappers around `calibrated-explanations` library and manual conformal
prediction implementations for static, online, and segment-level CP.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def create_calibrated_explainer(model, X_cal, y_cal, feature_names=None, **kwargs):
    """Initialize WrapCalibratedExplainer with a fitted model and calibration data.

    Uses the calibrated-explanations library's WrapCalibratedExplainer which
    wraps a pre-fitted model and calibrates it with held-out data.

    Parameters
    ----------
    model : fitted model
        A trained scikit-learn compatible model (e.g., XGBoost)
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    feature_names : list, optional
        Feature names for interpretability
    **kwargs
        Kept for backward compatibility; ignored.

    Returns
    -------
    WrapCalibratedExplainer
        Calibrated explainer ready for prediction
    """
    from calibrated_explanations import WrapCalibratedExplainer

    # WrapCalibratedExplainer wraps a pre-fitted model
    # mode is inferred automatically from the model, so we don't pass kwargs
    ce = WrapCalibratedExplainer(model)

    # fit() registers the model; calibrate() computes nonconformity scores
    #ce.fit(X_cal, y_cal)
    ce.calibrate(X_cal, y_cal, feature_names=feature_names)

    return ce


def _confidence_to_percentiles(confidence: float) -> tuple:
    """Convert a confidence level to low/high percentiles.

    E.g., confidence=0.90 -> (5, 95) for a 90% interval.
    """
    alpha = (1.0 - confidence) / 2.0
    low = alpha * 100
    high = (1.0 - alpha) * 100
    return (low, high)


def get_static_prediction_intervals(
    explainer,
    X_test,
    confidence: float = 0.90
):
    """Generate prediction intervals using static conformal prediction.

    Uses explain_factual for small datasets (with feature explanations).
    For large datasets, use get_fast_prediction_intervals instead.

    Parameters
    ----------
    explainer : WrapCalibratedExplainer
        Calibrated explainer
    X_test : np.ndarray
        Test features
    confidence : float
        Target coverage probability (e.g., 0.90 for 90%)

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays
    """
    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Get factual explanations with specified percentile bounds
    explanations = explainer.explain_factual(
        X_test,
        low_high_percentiles=(low_pct, high_pct)
    )

    # Extract predictions and intervals from the explanation object
    y_pred = []
    lower = []
    upper = []

    for exp in explanations:
        prediction = exp.prediction
        y_pred.append(prediction['predict'])
        lower.append(prediction['low'])
        upper.append(prediction['high'])

    return np.asarray(y_pred), np.asarray(lower), np.asarray(upper)


def get_fast_prediction_intervals(
    model,
    X_cal,
    y_cal,
    X_test,
    confidence: float = 0.90
):
    """Fast bulk prediction intervals using split conformal prediction.

    Computes nonconformity scores on calibration set, takes the quantile,
    and applies symmetric intervals to all test predictions. This is O(n)
    and suitable for large datasets (100K+ samples).

    Parameters
    ----------
    model : fitted model
        Trained model with .predict() method
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    X_test : np.ndarray
        Test features
    confidence : float
        Target coverage probability (e.g., 0.90 for 90%)

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays
    """
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    X_test = np.asarray(X_test)

    # Step 1: Compute nonconformity scores on calibration set
    y_cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred)

    # Step 2: Get the quantile at the desired confidence level
    # For finite-sample validity: quantile at ceil((n+1)*(1-alpha))/n
    n = len(residuals)
    alpha = 1.0 - confidence
    q_level = min(np.ceil((n + 1) * confidence) / n, 1.0)
    q = np.quantile(residuals, q_level)

    # Step 3: Generate predictions and intervals
    y_pred = model.predict(X_test)
    lower = y_pred - q
    upper = y_pred + q

    return y_pred, lower, upper


def get_normalized_prediction_intervals(
    model,
    X_cal,
    y_cal,
    X_test,
    segment_ids_cal=None,
    segment_ids_test=None,
    confidence: float = 0.90,
    min_samples_per_group: int = 30,
    fallback_to_global: bool = True
):
    """Normalized conformal prediction with per-segment adaptive widths.

    Uses Normalized Conformal Prediction (NCP): nonconformity scores are
    divided by a per-segment difficulty estimate (MAD of residuals), so that
    easy-to-predict segments get narrower intervals and hard segments get wider.

    Parameters
    ----------
    model : fitted model
        Trained model with .predict() method
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration target values
    X_test : np.ndarray
        Test features
    segment_ids_cal : np.ndarray, optional
        Segment identifier for each calibration sample (e.g., segment number).
        If None, uses a difficulty model (MAD of residuals from k-NN).
    segment_ids_test : np.ndarray, optional
        Segment identifier for each test sample
    confidence : float
        Target coverage probability
    min_samples_per_group : int
        Minimum calibration samples per segment to use group-specific sigma
    fallback_to_global : bool
        If a segment has too few calibration samples, fall back to global sigma

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays — widths vary per sample
    """
    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)
    X_test = np.asarray(X_test)

    # Step 1: Calibration residuals
    y_cal_pred = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - y_cal_pred)

    # Step 2: Estimate per-segment difficulty (sigma)
    if segment_ids_cal is not None and segment_ids_test is not None:
        segment_ids_cal = np.asarray(segment_ids_cal)
        segment_ids_test = np.asarray(segment_ids_test)

        # Compute MAD (median absolute deviation) of residuals per segment
        unique_segments = np.unique(segment_ids_cal)
        segment_sigma = {}
        global_sigma = np.median(cal_residuals) + 1e-6  # avoid division by zero

        for seg in unique_segments:
            mask = segment_ids_cal == seg
            if mask.sum() >= min_samples_per_group:
                segment_sigma[seg] = np.median(cal_residuals[mask]) + 1e-6
            elif fallback_to_global:
                segment_sigma[seg] = global_sigma

        # Assign sigma to calibration samples
        sigma_cal = np.array([segment_sigma.get(s, global_sigma) for s in segment_ids_cal])

        # Assign sigma to test samples
        sigma_test = np.array([segment_sigma.get(s, global_sigma) for s in segment_ids_test])
    else:
        # Without segment IDs, use model residual magnitude as difficulty proxy
        # Estimate difficulty via absolute prediction value (larger predictions = more uncertainty)
        y_test_pred = model.predict(X_test)

        # Use calibration residuals grouped by prediction magnitude quintiles
        cal_pred_abs = np.abs(y_cal_pred)
        quintiles = np.percentile(cal_pred_abs, [20, 40, 60, 80])

        def get_sigma(pred_val, cal_preds, cal_res):
            bin_idx = np.digitize(pred_val, quintiles)
            mask = np.digitize(cal_preds, quintiles) == bin_idx
            if mask.sum() >= min_samples_per_group:
                return np.median(cal_res[mask]) + 1e-6
            return np.median(cal_res) + 1e-6

        sigma_cal = np.array([get_sigma(p, cal_pred_abs, cal_residuals) for p in cal_pred_abs])
        sigma_test = np.array([get_sigma(p, cal_pred_abs, cal_residuals) for p in np.abs(y_test_pred)])

    # Step 3: Compute normalized nonconformity scores
    normalized_scores = cal_residuals / sigma_cal

    # Step 4: Get quantile of normalized scores
    n = len(normalized_scores)
    alpha = 1.0 - confidence
    q_level = min(np.ceil((n + 1) * confidence) / n, 1.0)
    q = np.quantile(normalized_scores, q_level)

    # Step 5: Generate predictions and adaptive intervals
    y_pred = model.predict(X_test)
    lower = y_pred - q * sigma_test
    upper = y_pred + q * sigma_test

    return y_pred, lower, upper


def get_online_prediction_intervals(
    model,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_cal_init: np.ndarray,
    y_cal_init: np.ndarray,
    confidence: float = 0.90,
    update_frequency: int = 1,
    window_size: int = None,
    dates_stream: np.ndarray = None,
    group_keys_stream: np.ndarray = None,
    verbose: bool = True
):
    """Online/adaptive conformal prediction with sequential calibration updates.

    After each prediction batch (defined by update_frequency), revealed true
    values are added to the calibration set, and the explainer is re-created.

    Parameters
    ----------
    model : fitted model
        Trained model (not retrained, only calibration updates)
    X_stream : np.ndarray
        Test features in temporal order
    y_stream : np.ndarray
        True test values (revealed after prediction)
    X_cal_init : np.ndarray
        Initial calibration features
    y_cal_init : np.ndarray
        Initial calibration targets
    confidence : float
        Target coverage
    update_frequency : int
        Number of samples between calibration updates.
        If dates_stream is provided, updates happen daily regardless.
    window_size : int, optional
        If None, use expanding window. If int, use sliding window of this many
        recent samples.
    dates_stream : np.ndarray, optional
        Dates for each sample. If provided, updates happen at each new day.
    group_keys_stream : array-like, optional
        Custom grouping keys for each sample (e.g., (date, hour) tuples).
        If provided, updates happen at each unique key. Takes precedence
        over dates_stream.
    verbose : bool
        Whether to show progress bar

    Returns
    -------
    tuple
        (y_pred_all, lower_all, upper_all, running_coverages)
    """
    from calibrated_explanations import WrapCalibratedExplainer

    # Convert to numpy arrays to avoid DataFrame indexing issues
    X_stream = np.asarray(X_stream)
    y_stream = np.asarray(y_stream)
    X_cal = np.copy(np.asarray(X_cal_init))
    y_cal = np.copy(np.asarray(y_cal_init))

    y_pred_all = []
    lower_all = []
    upper_all = []
    running_coverages = []

    n_total = len(X_stream)
    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Determine update points: group_keys_stream takes precedence over dates_stream
    keys_stream = group_keys_stream if group_keys_stream is not None else dates_stream

    if keys_stream is not None:
        unique_keys = sorted(set(keys_stream))
        key_to_indices = {}
        for i, k in enumerate(keys_stream):
            if k not in key_to_indices:
                key_to_indices[k] = []
            key_to_indices[k].append(i)

        desc = 'Online CP (by key)' if group_keys_stream is not None else 'Online CP (daily)'
        iterator = tqdm(unique_keys, desc=desc) if verbose else unique_keys

        for key in iterator:
            indices = key_to_indices[key]
            X_batch = X_stream[indices]
            y_batch = y_stream[indices]

            # Create explainer with current calibration set
            ce = WrapCalibratedExplainer(model)
            ce.calibrate(X_cal, y_cal)

            # Predict with intervals
            explanations = ce.explain_factual(
                X_batch, low_high_percentiles=(low_pct, high_pct)
            )

            for exp in explanations:
                pred = exp.prediction
                y_pred_all.append(pred['predict'])
                lower_all.append(pred['low'])
                upper_all.append(pred['high'])

            # Update calibration set
            X_cal = np.vstack([X_cal, X_batch])
            y_cal = np.concatenate([y_cal, y_batch])

            # Apply sliding window if specified
            if window_size is not None and len(y_cal) > window_size:
                X_cal = X_cal[-window_size:]
                y_cal = y_cal[-window_size:]

            # Track running coverage
            covered = np.array([(yt >= lo and yt <= up)
                               for yt, lo, up in zip(y_pred_all, lower_all, upper_all)])
            running_coverages.append(np.mean(covered))
    else:
        # Update by sample count
        ce = WrapCalibratedExplainer(model)
        ce.calibrate(X_cal, y_cal)

        iterator = tqdm(range(0, n_total, update_frequency),
                       desc='Online CP') if verbose else range(0, n_total, update_frequency)

        for start in iterator:
            end = min(start + update_frequency, n_total)
            X_batch = X_stream[start:end]
            y_batch = y_stream[start:end]

            # Predict with intervals
            explanations = ce.explain_factual(
                X_batch, low_high_percentiles=(low_pct, high_pct)
            )

            for exp in explanations:
                pred = exp.prediction
                y_pred_all.append(pred['predict'])
                lower_all.append(pred['low'])
                upper_all.append(pred['high'])

            # Update calibration set
            X_cal = np.vstack([X_cal, X_batch])
            y_cal = np.concatenate([y_cal, y_batch])

            if window_size is not None and len(y_cal) > window_size:
                X_cal = X_cal[-window_size:]
                y_cal = y_cal[-window_size:]

            # Re-create explainer with updated calibration
            ce = WrapCalibratedExplainer(model)
            ce.calibrate(X_cal, y_cal)

            covered = np.array([(yt >= lo and yt <= up)
                               for yt, lo, up in zip(y_pred_all, lower_all, upper_all)])
            running_coverages.append(np.mean(covered))

    return (
        np.asarray(y_pred_all),
        np.asarray(lower_all),
        np.asarray(upper_all),
        running_coverages
    )


def get_segment_level_intervals(
    explainer,
    X_test_by_segment: dict,
    confidence: float = 0.90
) -> dict:
    """Get prediction intervals per segment.

    Parameters
    ----------
    explainer : WrapCalibratedExplainer
        Calibrated explainer for the segment model
    X_test_by_segment : dict
        segment_id -> X_test array
    confidence : float
        Target coverage

    Returns
    -------
    dict
        segment_id -> (y_pred, lower, upper)
    """
    low_pct, high_pct = _confidence_to_percentiles(confidence)
    results = {}

    for seg_id, X_test in tqdm(X_test_by_segment.items(), desc='Segment intervals'):
        if len(X_test) == 0:
            continue

        explanations = explainer.explain_factual(
            X_test, low_high_percentiles=(low_pct, high_pct)
        )

        y_pred = []
        lower = []
        upper = []
        for exp in explanations:
            pred = exp.prediction
            y_pred.append(pred['predict'])
            lower.append(pred['low'])
            upper.append(pred['high'])

        results[seg_id] = (
            np.asarray(y_pred),
            np.asarray(lower),
            np.asarray(upper)
        )

    return results


def aggregate_segment_intervals_to_route(
    segment_intervals: dict,
    trip_segment_mapping: pd.DataFrame,
    method: str = 'sum'
):
    """Aggregate segment-level intervals to route-level.

    Parameters
    ----------
    segment_intervals : dict
        segment_id -> (y_pred, lower, upper) for each segment record
    trip_segment_mapping : pd.DataFrame
        DataFrame with trip_id, segment columns to map segments to trips
    method : str
        'sum' for simple summation, 'bonferroni' for Bonferroni-corrected

    Returns
    -------
    pd.DataFrame
        Route-level results: trip_id, y_pred_route, lower_route, upper_route
    """
    results = []

    for trip_id, trip_group in trip_segment_mapping.groupby('trip_id'):
        trip_pred = 0.0
        trip_lower = 0.0
        trip_upper = 0.0
        trip_actual = 0.0
        valid = True

        for _, row in trip_group.iterrows():
            seg_id = row['segment']
            if seg_id in segment_intervals:
                y_pred, lower, upper = segment_intervals[seg_id]
                # Find the corresponding index for this trip's segment
                # This needs to be matched by the order in X_test_by_segment
                idx = row.get('seg_test_idx', 0)
                if idx < len(y_pred):
                    trip_pred += y_pred[idx]
                    trip_lower += lower[idx]
                    trip_upper += upper[idx]
                else:
                    valid = False
                    break
            else:
                valid = False
                break

        if valid:
            results.append({
                'trip_id': trip_id,
                'y_pred_route': trip_pred,
                'lower_route': trip_lower,
                'upper_route': trip_upper,
            })

    return pd.DataFrame(results)


def compute_segment_uncertainty_attribution(
    segment_intervals: dict,
    trip_segments: list
) -> pd.DataFrame:
    """Compute each segment's contribution to total route uncertainty.

    Parameters
    ----------
    segment_intervals : dict
        segment_id -> (y_pred, lower, upper)
    trip_segments : list
        Ordered list of segment IDs in the trip

    Returns
    -------
    pd.DataFrame
        segment, width, fraction_of_total columns
    """
    rows = []
    total_width = 0.0

    for seg_id in trip_segments:
        if seg_id in segment_intervals:
            y_pred, lower, upper = segment_intervals[seg_id]
            # Use mean width across all samples for this segment
            width = np.mean(upper - lower)
        else:
            width = 0.0

        total_width += width
        rows.append({'segment': seg_id, 'width': width})

    df = pd.DataFrame(rows)
    df['fraction_of_total'] = df['width'] / total_width if total_width > 0 else 0
    df['cumulative_width'] = df['width'].cumsum()

    return df


# =============================================================================
# Mondrian (Conditional) Conformal Prediction
# =============================================================================

# Mondrian bins that have sufficient calibration samples
MONDRIAN_VALID_TIME_PERIODS = [
    'early_morning', 'morning_peak', 'midday', 'evening_peak', 'evening'
]

# Night period excluded due to insufficient calibration data (≤2 samples)
MONDRIAN_FALLBACK_PERIOD = 'night'


def assign_mondrian_bins(time_periods, route_names):
    """Assign Mondrian bin IDs from (time_period, route) combinations.

    Night-period samples are assigned bin_id = -1 (fallback to global).
    Valid bins are assigned sequential integer IDs.

    Parameters
    ----------
    time_periods : array-like
        Time period labels for each sample (e.g., 'morning_peak')
    route_names : array-like
        Route identifiers for each sample (e.g., 10, 12, 46)

    Returns
    -------
    bins : np.ndarray of int
        Bin ID per sample (-1 = fallback/night)
    bin_labels : dict
        Mapping from bin_id -> (time_period, route) human-readable label
    """
    time_periods = np.asarray(time_periods)
    route_names = np.asarray(route_names)

    # Build mapping: (time_period, route) -> int bin_id
    unique_routes = sorted(set(route_names))
    bin_labels = {}
    label_to_id = {}
    next_id = 0

    for tp in MONDRIAN_VALID_TIME_PERIODS:
        for route in unique_routes:
            key = (tp, route)
            bin_labels[next_id] = key
            label_to_id[key] = next_id
            next_id += 1

    # Assign bin IDs
    bins = np.full(len(time_periods), -1, dtype=int)
    for i in range(len(time_periods)):
        key = (time_periods[i], route_names[i])
        if key in label_to_id:
            bins[i] = label_to_id[key]

    return bins, bin_labels


def get_mondrian_bin_stats(bins, y_true=None):
    """Compute sample counts per Mondrian bin.

    Parameters
    ----------
    bins : np.ndarray
        Bin assignments (-1 = fallback)
    y_true : np.ndarray, optional
        Target values for summary statistics

    Returns
    -------
    pd.DataFrame
        Statistics per bin
    """
    unique_bins = np.unique(bins)
    rows = []
    for b in unique_bins:
        mask = bins == b
        row = {'bin_id': b, 'n_samples': mask.sum()}
        if y_true is not None:
            row['mean_y'] = np.mean(y_true[mask])
            row['std_y'] = np.std(y_true[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def create_mondrian_calibrated_explainer(
    model, X_cal, y_cal, bins_cal, feature_names=None
):
    """Create a CalibratedExplainer with Mondrian (per-bin) calibration.

    Parameters
    ----------
    model : fitted model
        Trained model with .predict()
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration targets
    bins_cal : np.ndarray
        Mondrian bin IDs for calibration samples (must not contain -1;
        filter out fallback samples before calling)
    feature_names : list, optional
        Feature names

    Returns
    -------
    WrapCalibratedExplainer
        Calibrated with per-bin nonconformity scores
    """
    from calibrated_explanations import WrapCalibratedExplainer

    ce = WrapCalibratedExplainer(model)
    ce.calibrate(X_cal, y_cal, feature_names=feature_names, bins=bins_cal)
    return ce


def get_mondrian_prediction_intervals(
    explainer_mondrian,
    explainer_global,
    X_test,
    bins_test,
    confidence: float = 0.90
):
    """Generate prediction intervals using Mondrian CP with global fallback.

    Samples with valid bins use the Mondrian explainer (per-bin quantiles).
    Samples with bin_id == -1 (e.g., night period) use the global explainer.

    Parameters
    ----------
    explainer_mondrian : WrapCalibratedExplainer
        Mondrian-calibrated explainer (with bins)
    explainer_global : WrapCalibratedExplainer
        Global-calibrated explainer (without bins, for fallback)
    X_test : np.ndarray
        Test features
    bins_test : np.ndarray
        Mondrian bin IDs for test samples (-1 = use global fallback)
    confidence : float
        Target coverage

    Returns
    -------
    tuple
        (y_pred, lower, upper) arrays
    """
    X_test = np.asarray(X_test)
    bins_test = np.asarray(bins_test)

    y_pred = np.zeros(len(X_test))
    lower = np.zeros(len(X_test))
    upper = np.zeros(len(X_test))

    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Mondrian samples (valid bins)
    valid_mask = bins_test >= 0
    if valid_mask.sum() > 0:
        X_valid = X_test[valid_mask]
        bins_valid = bins_test[valid_mask]

        explanations = explainer_mondrian.explain_factual(
            X_valid,
            low_high_percentiles=(low_pct, high_pct),
            bins=bins_valid
        )

        idx = 0
        for exp in explanations:
            pred = exp.prediction
            valid_indices = np.where(valid_mask)[0]
            y_pred[valid_indices[idx]] = pred['predict']
            lower[valid_indices[idx]] = pred['low']
            upper[valid_indices[idx]] = pred['high']
            idx += 1

    # Fallback samples (night / unknown bins)
    fallback_mask = bins_test < 0
    if fallback_mask.sum() > 0:
        X_fallback = X_test[fallback_mask]

        explanations = explainer_global.explain_factual(
            X_fallback,
            low_high_percentiles=(low_pct, high_pct)
        )

        idx = 0
        for exp in explanations:
            pred = exp.prediction
            fallback_indices = np.where(fallback_mask)[0]
            y_pred[fallback_indices[idx]] = pred['predict']
            lower[fallback_indices[idx]] = pred['low']
            upper[fallback_indices[idx]] = pred['high']
            idx += 1

    return y_pred, lower, upper


def get_online_mondrian_prediction_intervals(
    model,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    bins_stream: np.ndarray,
    X_cal_init: np.ndarray,
    y_cal_init: np.ndarray,
    bins_cal_init: np.ndarray,
    confidence: float = 0.90,
    window_size: int = None,
    dates_stream: np.ndarray = None,
    verbose: bool = True
):
    """Online Mondrian CP with sequential calibration updates.

    Combines online recalibration (expanding/sliding window) with
    Mondrian binning (per-category quantiles). Night-period samples
    use global fallback at each update step.

    Parameters
    ----------
    model : fitted model
        Trained model (not retrained)
    X_stream : np.ndarray
        Test features in temporal order
    y_stream : np.ndarray
        True test values (revealed after prediction)
    bins_stream : np.ndarray
        Mondrian bin IDs for stream (-1 = fallback)
    X_cal_init : np.ndarray
        Initial calibration features
    y_cal_init : np.ndarray
        Initial calibration targets
    bins_cal_init : np.ndarray
        Initial calibration bin IDs (only valid bins, no -1)
    confidence : float
        Target coverage
    window_size : int, optional
        None = expanding, int = sliding window size
    dates_stream : np.ndarray, optional
        Dates for daily update batching
    verbose : bool
        Show progress bar

    Returns
    -------
    tuple
        (y_pred_all, lower_all, upper_all, running_coverages)
    """
    from calibrated_explanations import WrapCalibratedExplainer

    X_stream = np.asarray(X_stream)
    y_stream = np.asarray(y_stream)
    bins_stream = np.asarray(bins_stream)
    X_cal = np.copy(np.asarray(X_cal_init))
    y_cal = np.copy(np.asarray(y_cal_init))
    bins_cal = np.copy(np.asarray(bins_cal_init))

    y_pred_all = np.zeros(len(X_stream))
    lower_all = np.zeros(len(X_stream))
    upper_all = np.zeros(len(X_stream))
    running_coverages = []

    low_pct, high_pct = _confidence_to_percentiles(confidence)

    # Determine update points
    if dates_stream is not None:
        dates_stream = np.asarray(dates_stream)
        unique_keys = sorted(set(dates_stream))
        key_to_indices = {}
        for i, k in enumerate(dates_stream):
            key_to_indices.setdefault(k, []).append(i)
    else:
        unique_keys = [0]
        key_to_indices = {0: list(range(len(X_stream)))}

    desc = 'Online Mondrian CP (daily)' if dates_stream is not None else 'Online Mondrian CP'
    iterator = tqdm(unique_keys, desc=desc) if verbose else unique_keys

    for key in iterator:
        indices = key_to_indices[key]
        X_batch = X_stream[indices]
        y_batch = y_stream[indices]
        bins_batch = bins_stream[indices]

        # Separate valid (Mondrian) and fallback samples in the batch
        valid_mask_batch = bins_batch >= 0
        fallback_mask_batch = bins_batch < 0

        # --- Build Mondrian explainer from current calibration ---
        # Only use valid-bin calibration samples for Mondrian
        valid_cal_mask = bins_cal >= 0
        X_cal_valid = X_cal[valid_cal_mask]
        y_cal_valid = y_cal[valid_cal_mask]
        bins_cal_valid = bins_cal[valid_cal_mask]

        ce_mondrian = WrapCalibratedExplainer(model)
        ce_mondrian.calibrate(X_cal_valid, y_cal_valid, bins=bins_cal_valid)

        # --- Build global explainer for fallback ---
        ce_global = WrapCalibratedExplainer(model)
        ce_global.calibrate(X_cal, y_cal)

        # --- Predict valid-bin samples ---
        if valid_mask_batch.sum() > 0:
            valid_indices = [indices[j] for j in range(len(indices)) if valid_mask_batch[j]]
            X_valid = X_batch[valid_mask_batch]
            bins_valid = bins_batch[valid_mask_batch]

            explanations = ce_mondrian.explain_factual(
                X_valid,
                low_high_percentiles=(low_pct, high_pct),
                bins=bins_valid
            )
            for idx_pos, exp in enumerate(explanations):
                pred = exp.prediction
                gi = valid_indices[idx_pos]
                y_pred_all[gi] = pred['predict']
                lower_all[gi] = pred['low']
                upper_all[gi] = pred['high']

        # --- Predict fallback samples ---
        if fallback_mask_batch.sum() > 0:
            fallback_indices = [indices[j] for j in range(len(indices)) if fallback_mask_batch[j]]
            X_fb = X_batch[fallback_mask_batch]

            explanations = ce_global.explain_factual(
                X_fb,
                low_high_percentiles=(low_pct, high_pct)
            )
            for idx_pos, exp in enumerate(explanations):
                pred = exp.prediction
                gi = fallback_indices[idx_pos]
                y_pred_all[gi] = pred['predict']
                lower_all[gi] = pred['low']
                upper_all[gi] = pred['high']

        # --- Update calibration set ---
        X_cal = np.vstack([X_cal, X_batch])
        y_cal = np.concatenate([y_cal, y_batch])
        bins_cal = np.concatenate([bins_cal, bins_batch])

        if window_size is not None and len(y_cal) > window_size:
            X_cal = X_cal[-window_size:]
            y_cal = y_cal[-window_size:]
            bins_cal = bins_cal[-window_size:]

        # Running coverage
        processed = np.where(
            (y_pred_all != 0) | (lower_all != 0) | (upper_all != 0)
        )[0]
        if len(processed) > 0:
            covered = ((y_stream[processed] >= lower_all[processed]) &
                       (y_stream[processed] <= upper_all[processed]))
            running_coverages.append(np.mean(covered))

    return y_pred_all, lower_all, upper_all, running_coverages


# =============================================================================
# Normalized CP with DifficultyEstimator (crepes)
# =============================================================================

def create_difficulty_estimator(X_cal, y_cal, model, k=25, scaler=True, beta=0.01):
    """Create and fit a DifficultyEstimator using KNN on calibration residuals.

    The estimator learns how "difficult" each region of the feature space is
    based on the calibration residuals. At prediction time, it assigns a
    per-sample sigma that scales the conformal interval width.

    Parameters
    ----------
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration targets
    model : fitted model
        Trained model with .predict()
    k : int
        Number of neighbors for KNN difficulty estimation
    scaler : bool
        If True, normalize sigmas to [0, 1] range (+ beta)
    beta : float
        Small offset added to avoid division by zero

    Returns
    -------
    DifficultyEstimator
        Fitted estimator ready for .apply(X_test)
    """
    from crepes.extras import DifficultyEstimator

    X_cal = np.asarray(X_cal)
    y_cal = np.asarray(y_cal)

    # Compute calibration residuals
    y_cal_pred = model.predict(X_cal)
    residuals_cal = np.abs(y_cal - y_cal_pred)

    de = DifficultyEstimator()
    de.fit(X=X_cal, residuals=residuals_cal, k=k, scaler=scaler, beta=beta)
    return de


def create_calibrated_explainer_with_difficulty(
    model, X_cal, y_cal, difficulty_estimator, feature_names=None
):
    """Create a CalibratedExplainer with difficulty-based normalized CP.

    Parameters
    ----------
    model : fitted model
        Trained model
    X_cal : np.ndarray
        Calibration features
    y_cal : np.ndarray
        Calibration targets
    difficulty_estimator : DifficultyEstimator
        Fitted difficulty estimator from crepes
    feature_names : list, optional
        Feature names

    Returns
    -------
    WrapCalibratedExplainer
        Calibrated with per-sample adaptive intervals
    """
    from calibrated_explanations import WrapCalibratedExplainer

    ce = WrapCalibratedExplainer(model)
    ce.calibrate(X_cal, y_cal, feature_names=feature_names,
                 difficulty_estimator=difficulty_estimator)
    return ce


# =============================================================================
# Adaptive Conformal Inference (ACI) — Gibbs & Candès, 2021
# =============================================================================

def get_aci_prediction_intervals(
    model,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    X_cal_init: np.ndarray,
    y_cal_init: np.ndarray,
    target_coverage: float = 0.90,
    gamma: float = 0.005,
    window_size: int = None,
    dates_stream: np.ndarray = None,
    verbose: bool = True
):
    """Adaptive Conformal Inference with online calibration.

    ACI dynamically adjusts the miscoverage rate alpha_t based on observed
    coverage errors, providing long-run coverage guarantees WITHOUT requiring
    the exchangeability assumption.

    After each sample (or daily batch), alpha is updated:
        alpha_{t+1} = alpha_t + gamma * (alpha_t - err_t)
    where err_t = 1 if sample was missed, 0 if covered.

    This is combined with online calibration (expanding or sliding window)
    so that both the quantile level AND the calibration set adapt over time.

    Parameters
    ----------
    model : fitted model
        Trained model with .predict()
    X_stream : np.ndarray
        Test features in temporal order
    y_stream : np.ndarray
        True values (revealed after prediction)
    X_cal_init : np.ndarray
        Initial calibration features (e.g., W4)
    y_cal_init : np.ndarray
        Initial calibration targets
    target_coverage : float
        Target coverage (e.g., 0.90)
    gamma : float
        ACI learning rate (step size). Larger gamma = faster adaptation
        but more volatile. Typical range: 0.001 to 0.05.
    window_size : int, optional
        None = expanding window, int = sliding window
    dates_stream : np.ndarray, optional
        If provided, update in daily batches
    verbose : bool
        Show progress bar

    Returns
    -------
    dict with keys:
        y_pred, lower, upper : np.ndarray — predictions and intervals
        alpha_history : list — alpha_t at each update step
        coverage_history : list — running coverage at each step
        width_history : list — mean interval width at each step
    """
    X_stream = np.asarray(X_stream)
    y_stream = np.asarray(y_stream)
    X_cal = np.copy(np.asarray(X_cal_init))
    y_cal = np.copy(np.asarray(y_cal_init))

    n_total = len(X_stream)
    y_pred_all = np.zeros(n_total)
    lower_all = np.zeros(n_total)
    upper_all = np.zeros(n_total)

    # ACI state
    alpha_t = 1.0 - target_coverage  # initial alpha = 0.10
    alpha_history = []
    coverage_history = []
    width_history = []

    # Determine update batches
    if dates_stream is not None:
        dates_stream = np.asarray(dates_stream)
        unique_keys = sorted(set(dates_stream))
        key_to_indices = {}
        for i, k in enumerate(dates_stream):
            key_to_indices.setdefault(k, []).append(i)
    else:
        unique_keys = list(range(n_total))
        key_to_indices = {i: [i] for i in range(n_total)}

    iterator = tqdm(unique_keys, desc='ACI') if verbose else unique_keys

    for key in iterator:
        indices = key_to_indices[key]
        X_batch = X_stream[indices]
        y_batch = y_stream[indices]

        # --- Compute calibration residuals and quantile at current alpha ---
        y_cal_pred = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - y_cal_pred)

        # Clamp alpha to valid range
        alpha_clamped = np.clip(alpha_t, 0.001, 0.999)
        confidence_t = 1.0 - alpha_clamped

        # Finite-sample corrected quantile level
        n_cal = len(cal_residuals)
        q_level = min(np.ceil((n_cal + 1) * confidence_t) / n_cal, 1.0)
        q_t = np.quantile(cal_residuals, q_level)

        # --- Predict with current q ---
        y_pred_batch = model.predict(X_batch)
        lower_batch = y_pred_batch - q_t
        upper_batch = y_pred_batch + q_t

        for j, gi in enumerate(indices):
            y_pred_all[gi] = y_pred_batch[j]
            lower_all[gi] = lower_batch[j]
            upper_all[gi] = upper_batch[j]

        # --- Compute batch error rate ---
        covered_batch = ((y_batch >= lower_batch) & (y_batch <= upper_batch))
        err_batch = 1.0 - np.mean(covered_batch)  # fraction missed

        # --- ACI update: alpha_{t+1} = alpha_t + gamma * (alpha_t - err_t) ---
        # When err > alpha (too many misses): alpha decreases → wider intervals
        # When err < alpha (too few misses): alpha increases → narrower intervals
        alpha_t = alpha_t + gamma * (alpha_t - err_batch)

        # --- Update calibration set ---
        X_cal = np.vstack([X_cal, X_batch])
        y_cal = np.concatenate([y_cal, y_batch])

        if window_size is not None and len(y_cal) > window_size:
            X_cal = X_cal[-window_size:]
            y_cal = y_cal[-window_size:]

        # --- Track history ---
        alpha_history.append(float(alpha_clamped))
        processed_mask = (y_pred_all != 0) | (lower_all != 0) | (upper_all != 0)
        if processed_mask.any():
            proc_idx = np.where(processed_mask)[0]
            cov = np.mean((y_stream[proc_idx] >= lower_all[proc_idx]) &
                          (y_stream[proc_idx] <= upper_all[proc_idx]))
            coverage_history.append(float(cov))
        width_history.append(float(2 * q_t))

    return {
        'y_pred': y_pred_all,
        'lower': lower_all,
        'upper': upper_all,
        'alpha_history': alpha_history,
        'coverage_history': coverage_history,
        'width_history': width_history,
    }
