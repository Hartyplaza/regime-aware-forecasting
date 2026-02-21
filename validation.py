"""
Data validation and quality checks.
Ensures data integrity before feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass

from config import (
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
            report += f"Warnings:\n"
            for w in self.warnings:
                report += f"  - {w}\n"
        report += f"{'='*60}\n"
        return report


class DataValidator:
    """Production-grade data validation."""
    
    @staticmethod
    def validate_ohlcv(
        df: pd.DataFrame,
        min_records: int = MIN_DATA_POINTS,
        max_missing_pct: float = MAX_MISSING_PCT,
        outlier_threshold: float = OUTLIER_ZSCORE_THRESHOLD,
    ) -> ValidationReport:
        """
        Validate OHLCV data.
        
        Args:
            df: DataFrame with columns [Date, Open, High, Low, Close, Volume]
            min_records: Minimum required data points
            max_missing_pct: Tolerance for missing values
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            ValidationReport with findings
        """
        warnings = []
        outliers_detected = {}
        is_valid = True
        
        # Check basic structure
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        
        n_records = len(df)
        
        # Check minimum records
        if n_records < min_records:
            is_valid = False
            warnings.append(f"Insufficient records: {n_records} < {min_records}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        missing_pct = missing_count / (len(df) * len(df.columns))
        
        if missing_pct > max_missing_pct:
            is_valid = False
            warnings.append(f"Too many missing values: {missing_pct:.2%} > {max_missing_pct:.2%}")
        
        # Check for obvious pricing errors (OHLC logic)
        invalid_ohlc = (
            (df["Open"] <= 0) | (df["High"] <= 0) | 
            (df["Low"] <= 0) | (df["Close"] <= 0) |
            (df["High"] < df["Low"]) |
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"])
        )
        
        if invalid_ohlc.any():
            is_valid = False
            warnings.append(f"Invalid OHLC relationships detected in {invalid_ohlc.sum()} rows")
        
        # Check for volume anomalies
        if (df["Volume"] <= 0).any():
            is_valid = False
            warnings.append("Zero or negative volume detected")
        
        # Detect outliers in returns
        returns = df["Close"].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        high_outliers = (z_scores > outlier_threshold).sum()
        if high_outliers > 0:
            outliers_detected["returns"] = int(high_outliers)
            warnings.append(f"Outliers in returns: {high_outliers} observations with |z| > {outlier_threshold}")
        
        # Check date continuity (non-trading days are OK)
        date_gaps = df.index.to_series().diff()
        expected_gap = pd.Timedelta(days=1)
        multi_day_gaps = (date_gaps > expected_gap).sum()
        
        # Count missing trading days (rough estimate)
        trading_days_expected = len(pd.bdate_range(df.index[0], df.index[-1]))
        missing_dates = max(0, trading_days_expected - len(df))
        
        # Warn if unusual gaps exist (but don't fail on weekends/holidays)
        if multi_day_gaps > 10:  # Arbitrary threshold
            warnings.append(f"Unusual date gaps detected: {multi_day_gaps} multi-day gaps")
        
        return ValidationReport(
            is_valid=is_valid,
            n_records=n_records,
            missing_pct=missing_pct,
            missing_dates=missing_dates,
            outliers_detected=outliers_detected,
            warnings=warnings,
        )
    
    @staticmethod
    def forward_fill_missing(
        df: pd.DataFrame,
        max_fill_periods: int = 5,
    ) -> pd.DataFrame:
        """
        Forward fill missing OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            max_fill_periods: Max consecutive periods to fill
            
        Returns:
            Filled DataFrame
        """
        filled = df.copy()
        
        # Forward fill, but limit consecutive fills
        for col in df.columns:
            filled[col] = filled[col].fillna(method='ffill', limit=max_fill_periods)
        
        # Raise error if any NaN remains
        if filled.isnull().any().any():
            raise ValueError("Unable to fill all missing values within limit")
        
        return filled
