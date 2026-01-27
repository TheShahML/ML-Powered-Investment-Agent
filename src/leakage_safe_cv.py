"""
Leakage-Safe Cross-Validation for Time-Series Panel Data.

Implements date-grouped purged cross-validation to prevent:
1. Data leakage from overlapping dates between train/val
2. Lookahead bias from using future information
3. Label leakage from overlapping prediction horizons

Based on Advances in Financial Machine Learning (Lopez de Prado, 2018)
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Tuple
from datetime import timedelta


class DateGroupedPurgedCV:
    """
    Cross-validation that splits by date and adds embargo periods.

    Key features:
    - Splits on unique dates, not rows (prevents same-date leakage)
    - Adds embargo between train/val (accounts for prediction horizon)
    - Computes per-date cross-sectional metrics (not per-row)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 0,
        test_size_pct: float = 0.2,
        min_train_dates: int = 252  # 1 year minimum
    ):
        """
        Args:
            n_splits: Number of CV folds
            embargo_days: Days to embargo between train and validation
            test_size_pct: Fraction of dates for validation (per fold)
            min_train_dates: Minimum training dates required
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.test_size_pct = test_size_pct
        self.min_train_dates = min_train_dates

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CV splits based on dates.

        Args:
            X: Feature DataFrame with MultiIndex (timestamp, symbol) or DatetimeIndex
            y: Target series (optional, not used for splitting)

        Yields:
            (train_indices, val_indices) tuples
        """
        # Extract unique dates
        if isinstance(X.index, pd.MultiIndex):
            dates = pd.to_datetime(X.index.get_level_values(0)).unique().sort_values()
        else:
            dates = pd.to_datetime(X.index).unique().sort_values()

        dates = pd.Series(dates)
        n_dates = len(dates)

        logger.info(f"DateGroupedPurgedCV: {n_dates} unique dates, {self.n_splits} splits")
        logger.info(f"  Embargo: {self.embargo_days} days")
        logger.info(f"  Test size: {self.test_size_pct*100:.1f}%")

        # Calculate test size in number of dates
        test_dates = int(n_dates * self.test_size_pct)
        test_dates = max(test_dates, 21)  # At least ~1 month

        # Generate splits (walk-forward)
        for i in range(self.n_splits):
            # Validation period: moving window
            val_end_idx = n_dates - (self.n_splits - i - 1) * test_dates
            val_start_idx = val_end_idx - test_dates

            # Ensure we don't go negative
            if val_start_idx < self.min_train_dates:
                logger.warning(f"Fold {i+1}: Insufficient data, skipping")
                continue

            val_start_date = dates.iloc[val_start_idx]
            val_end_date = dates.iloc[val_end_idx - 1]

            # Embargo period: gap between train and val
            embargo_end_date = val_start_date - timedelta(days=self.embargo_days)

            # Training period: all dates before embargo
            train_dates_mask = dates < embargo_end_date
            train_indices = self._get_indices_for_dates(X, dates[train_dates_mask])

            # Validation period
            val_dates_mask = (dates >= val_start_date) & (dates <= val_end_date)
            val_indices = self._get_indices_for_dates(X, dates[val_dates_mask])

            if len(train_indices) < 1000:
                logger.warning(f"Fold {i+1}: Too few training samples ({len(train_indices)}), skipping")
                continue

            logger.info(
                f"Fold {i+1}/{self.n_splits}: "
                f"Train dates: {len(dates[train_dates_mask])} "
                f"(up to {embargo_end_date.date()}), "
                f"Val dates: {len(dates[val_dates_mask])} "
                f"({val_start_date.date()} to {val_end_date.date()}), "
                f"Samples: {len(train_indices)} train, {len(val_indices)} val"
            )

            yield train_indices, val_indices

    def _get_indices_for_dates(self, X: pd.DataFrame, date_mask: pd.Series) -> np.ndarray:
        """Get row indices corresponding to dates in date_mask."""
        if isinstance(X.index, pd.MultiIndex):
            # MultiIndex: filter by level 0 (timestamp)
            timestamps = pd.to_datetime(X.index.get_level_values(0))
            mask = timestamps.isin(date_mask)
        else:
            # Simple DatetimeIndex
            timestamps = pd.to_datetime(X.index)
            mask = timestamps.isin(date_mask)

        return np.where(mask)[0]


def compute_cross_sectional_ic(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute cross-sectional Information Coefficient (IC) by date.

    IC = Spearman rank correlation between predictions and realized returns
    on each date across all stocks.

    Args:
        y_true: True labels with MultiIndex (timestamp, symbol)
        y_pred: Predicted values (same length as y_true)

    Returns:
        Dict with IC statistics: mean_ic, std_ic, pct_positive_ic, ic_series
    """
    from scipy.stats import spearmanr

    if not isinstance(y_true.index, pd.MultiIndex):
        logger.warning("y_true does not have MultiIndex, computing global correlation")
        corr, _ = spearmanr(y_pred, y_true)
        return {
            'mean_ic': corr,
            'std_ic': 0.0,
            'pct_positive_ic': 100.0 if corr > 0 else 0.0,
            'ic_series': [corr]
        }

    # Create DataFrame for easier grouping
    df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred
    }, index=y_true.index)

    # Compute IC per date
    ic_by_date = []

    for date, group in df.groupby(level=0):
        if len(group) < 10:  # Need enough stocks for meaningful correlation
            continue

        try:
            corr, _ = spearmanr(group['y_pred'], group['y_true'])
            if not np.isnan(corr):
                ic_by_date.append(corr)
        except:
            continue

    if len(ic_by_date) == 0:
        return {
            'mean_ic': 0.0,
            'std_ic': 0.0,
            'pct_positive_ic': 0.0,
            'ic_series': []
        }

    ic_series = np.array(ic_by_date)

    return {
        'mean_ic': np.mean(ic_series),
        'std_ic': np.std(ic_series),
        'pct_positive_ic': 100.0 * np.mean(ic_series > 0),
        'ic_series': ic_series.tolist(),
        'n_dates': len(ic_series)
    }


def train_with_leakage_safe_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    embargo_days: int,
    n_splits: int = 5
) -> dict:
    """
    Train model with leakage-safe CV and return metrics.

    Args:
        model: Model instance with fit() and predict() methods
        X: Features (MultiIndex: timestamp, symbol)
        y: Target (MultiIndex: timestamp, symbol)
        embargo_days: Embargo period matching prediction horizon
        n_splits: Number of CV folds

    Returns:
        Dict with CV metrics and per-fold results
    """
    cv = DateGroupedPurgedCV(
        n_splits=n_splits,
        embargo_days=embargo_days,
        test_size_pct=0.2
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info(f"Training fold {fold_idx + 1}/{n_splits}...")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Remove NaN/inf
        train_mask = np.isfinite(X_train.values).all(axis=1) & np.isfinite(y_train.values)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        val_mask = np.isfinite(X_val.values).all(axis=1) & np.isfinite(y_val.values)
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        if len(X_train) < 100 or len(X_val) < 100:
            logger.warning(f"Fold {fold_idx + 1}: Insufficient clean data after NaN removal")
            continue

        # Train
        model.fit(X_train.values, y_train.values)

        # Predict
        y_pred = model.predict(X_val.values)

        # Compute IC
        ic_metrics = compute_cross_sectional_ic(y_val, y_pred)

        fold_results.append({
            'fold': fold_idx + 1,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            **ic_metrics
        })

        logger.info(
            f"  Fold {fold_idx + 1} IC: {ic_metrics['mean_ic']:.4f} "
            f"(±{ic_metrics['std_ic']:.4f}), "
            f"{ic_metrics['pct_positive_ic']:.1f}% positive days"
        )

    if len(fold_results) == 0:
        logger.error("No valid CV folds completed!")
        return {'mean_ic': 0.0, 'std_ic': 0.0, 'pct_positive_ic': 0.0, 'folds': []}

    # Aggregate across folds
    all_mean_ics = [f['mean_ic'] for f in fold_results]
    all_pct_pos = [f['pct_positive_ic'] for f in fold_results]

    summary = {
        'cv_mean_ic': np.mean(all_mean_ics),
        'cv_std_ic': np.std(all_mean_ics),
        'cv_min_ic': np.min(all_mean_ics),
        'cv_max_ic': np.max(all_mean_ics),
        'cv_mean_pct_positive': np.mean(all_pct_pos),
        'n_folds_completed': len(fold_results),
        'folds': fold_results
    }

    logger.info("=" * 60)
    logger.info(f"CV SUMMARY: Mean IC = {summary['cv_mean_ic']:.4f} ± {summary['cv_std_ic']:.4f}")
    logger.info(f"  Range: [{summary['cv_min_ic']:.4f}, {summary['cv_max_ic']:.4f}]")
    logger.info(f"  Avg % positive days: {summary['cv_mean_pct_positive']:.1f}%")
    logger.info("=" * 60)

    return summary
