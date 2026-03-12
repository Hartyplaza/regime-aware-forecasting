"""
Data validation and quality checks.
Ensures data integrity before feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

from .config import (
    MIN_DATA_POINTS,
    MAX_MISSING_PCT,
    OUTLIER_ZSCORE_THRESHOLD,
)


@dataclass
class ValidationReport:
    """Container for validation results."""
    is_valid: bool
    n_records: int
    missing_pct: float
    missing_dates: int
    outliers_detected: Dict[str, int]
    warnings: List[str]

    def __str__(self):
        report = f"\n{'='*60}\nVALIDATION REPORT\n{'='*60}\n"
        report += f"Valid: {self.is_valid}\n"
        report += f"Records: {self.n_records}\n"
        report += f"Missing: {self.missing_pct:.2%}\n"
        report += f"Missing Dates (non-trading): {self.missing_dates}\n"
        if self.outliers_detected:
            report += f"Outliers Detected: {self.outliers_detected}\n"
        if self.warnings:
            report += "Warnings:\n"
            for w in self.warnings:
                report += f"  - {w}\n"
        report += f"{'='*60}\n"
        return report


class DataValidator:
    """Production-grade data validation."""

    @staticmethod
    def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
        """
        Safely extract one column as a Series, even if duplicate column names
        or DataFrame slices appear.
        """
        out = df[col]
        if isinstance(out, pd.DataFrame):
            out = out.iloc[:, 0]
        return out

    @staticmethod
    def validate_ohlcv(
        df: pd.DataFrame,
        min_records: int = MIN_DATA_POINTS,
        max_missing_pct: float = MAX_MISSING_PCT,
        outlier_threshold: float = OUTLIER_ZSCORE_THRESHOLD,
    ) -> ValidationReport:
        warnings = []
        outliers_detected = {}
        is_valid = True

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")

        n_records = len(df)

        if n_records < min_records:
            is_valid = False
            warnings.append(f"Insufficient records: {n_records} < {min_records}")

        missing_count = df.isnull().sum().sum()
        missing_pct = missing_count / (len(df) * len(df.columns))

        if missing_pct > max_missing_pct:
            is_valid = False
            warnings.append(f"Too many missing values: {missing_pct:.2%} > {max_missing_pct:.2%}")

        open_ = DataValidator._as_series(df, "Open")
        high_ = DataValidator._as_series(df, "High")
        low_ = DataValidator._as_series(df, "Low")
        close_ = DataValidator._as_series(df, "Close")
        volume_ = DataValidator._as_series(df, "Volume")

        invalid_ohlc = (
            (open_ <= 0) | (high_ <= 0) |
            (low_ <= 0) | (close_ <= 0) |
            (high_ < low_) |
            (high_ < open_) |
            (high_ < close_) |
            (low_ > open_) |
            (low_ > close_)
        )

        invalid_count = int(invalid_ohlc.fillna(False).sum())
        if invalid_count > 0:
            is_valid = False
            warnings.append(f"Invalid OHLC relationships detected in {invalid_count} rows")

        bad_volume = int((volume_ <= 0).fillna(False).sum())
        if bad_volume > 0:
            is_valid = False
            warnings.append(f"Zero or negative volume detected in {bad_volume} rows")

        returns = close_.pct_change()
        returns_std = returns.std()

        if pd.notna(returns_std) and returns_std != 0:
            z_scores = np.abs((returns - returns.mean()) / returns_std)
            high_outliers = int((z_scores > outlier_threshold).fillna(False).sum())
            if high_outliers > 0:
                outliers_detected["returns"] = high_outliers
                warnings.append(
                    f"Outliers in returns: {high_outliers} observations with |z| > {outlier_threshold}"
                )

        date_gaps = df.index.to_series().diff()
        expected_gap = pd.Timedelta(days=1)
        multi_day_gaps = int((date_gaps > expected_gap).sum())

        trading_days_expected = len(pd.bdate_range(df.index[0], df.index[-1]))
        missing_dates = max(0, trading_days_expected - len(df))

        if multi_day_gaps > 10:
            warnings.append(f"Unusual date gaps detected: {multi_day_gaps} multi-day gaps")

        return ValidationReport(
            is_valid=is_valid,
            n_records=n_records,
            missing_pct=float(missing_pct),
            missing_dates=missing_dates,
            outliers_detected=outliers_detected,
            warnings=warnings,
        )

    @staticmethod
    def forward_fill_missing(
        df: pd.DataFrame,
        max_fill_periods: int = 5,
    ) -> pd.DataFrame:
        filled = df.copy()
        filled = filled.ffill(limit=max_fill_periods)

        if filled.isnull().any().any():
            raise ValueError("Unable to fill all missing values within limit")

        return filled
