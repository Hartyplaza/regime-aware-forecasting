"""
Data pipeline: ingestion, cleaning, and feature engineering.
Orchestrates all data preparation for downstream modeling.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

import yfinance as yf
from config import DataConfig, DIRECTION_FORWARD_WINDOW
from validation import DataValidator, ValidationReport

# Suppress yfinance FutureWarning about auto_adjust default
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """
    Immutable output container for the data pipeline.
    Ensures downstream code cannot accidentally corrupt data.
    """
    df: pd.DataFrame  # Main feature matrix
    raw_df: pd.DataFrame  # Original OHLCV data (for reference)
    config: DataConfig
    validation_report: ValidationReport
    feature_names: list  # Names of engineered features (excluding target)
    
    def __post_init__(self):
        """Freeze dataframes."""
        self.df = self.df.copy()
        self.raw_df = self.raw_df.copy()
        self.df.flags.writeable = False
        self.raw_df.flags.writeable = False
    
    def get_features(self, include_target: bool = False) -> pd.DataFrame:
        """
        Extract feature matrix.
        
        Args:
            include_target: If True, include 'direction' target column
            
        Returns:
            DataFrame with features (and optionally target)
        """
        cols = self.feature_names.copy()
        if include_target and "direction" in self.df.columns:
            cols.append("direction")
        return self.df[cols].copy()
    
    def summary(self) -> str:
        """Print summary statistics."""
        summary = f"\n{'='*70}\nPIPELINE OUTPUT SUMMARY\n{'='*70}\n"
        summary += f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n"
        summary += f"Date range: {self.df.index[0].date()} to {self.df.index[-1].date()}\n"
        summary += f"Features: {', '.join(self.feature_names)}\n"
        summary += f"Missing values: {self.df.isnull().sum().sum()}\n"
        summary += f"{'='*70}\n"
        return summary


class DataPipeline:
    """
    Production data pipeline for regime-aware forecasting.
    
    Workflow:
        1. Fetch raw OHLCV data
        2. Validate data integrity
        3. Clean missing values
        4. Engineer features
        5. Return immutable output
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: DataConfig object (uses defaults if None)
        """
        self.config = config or DataConfig()
        self.validator = DataValidator()
        self.logger = logger
    
    def run(self) -> PipelineOutput:
        """
        Execute the full pipeline.
        
        Returns:
            PipelineOutput with features, raw data, validation report
        """
        self.logger.info(f"Starting pipeline for {self.config.ticker}")
        
        # Step 1: Fetch data
        self.logger.info("Fetching data...")
        raw_df = self._fetch_data()
        
        # Step 2: Validate
        self.logger.info("Validating data...")
        validation_report = self.validator.validate_ohlcv(raw_df)
        self.logger.info(validation_report)
        
        if not validation_report.is_valid:
            raise ValueError("Data validation failed. See report above.")
        
        # Step 3: Clean
        self.logger.info("Cleaning data...")
        clean_df = self.validator.forward_fill_missing(raw_df)
        
        # Step 4: Engineer features
        self.logger.info("Engineering features...")
        feature_df = self._engineer_features(clean_df)
        
        # Step 5: Return
        self.logger.info("Pipeline complete.")
        
        return PipelineOutput(
            df=feature_df,
            raw_df=clean_df,
            config=self.config,
            validation_report=validation_report,
            feature_names=self._get_feature_names(),
        )
    
    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Returns:
            DataFrame with index=Date (UTC), columns=[Open, High, Low, Close, Volume]
        """
        df = yf.download(
            self.config.ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False,
        )
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Select only required columns (yfinance returns: Open, High, Low, Close, Adj Close, Volume)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features.
        
        Features computed:
        - log returns
        - lagged returns (multiple lags)
        - realized volatility (multiple windows)
        - direction label (forward-looking)
        
        Args:
            df: Clean OHLCV DataFrame
            
        Returns:
            DataFrame with original OHLCV + engineered features
        """
        features = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        
        # Log returns
        features["log_return"] = np.log(features["Close"] / features["Close"].shift(1))
        
        # Lagged returns
        for lag in self.config.lag_windows:
            features[f"return_lag_{lag}"] = features["log_return"].shift(lag)
        
        # Realized volatility (rolling std of returns)
        for window in self.config.volatility_windows:
            features[f"volatility_{window}"] = (
                features["log_return"]
                .rolling(window=window)
                .std()
            )
        
        # High-Low spread (intraday volatility proxy)
        features["hl_spread"] = np.log(features["High"] / features["Low"])
        
        # Volume features
        features["volume_ma_ratio"] = (
            features["Volume"] / features["Volume"].rolling(window=20).mean()
        )
        
        # Direction label (forward-looking, 1 if next period return > 0 else 0)
        features["direction"] = (
            features["log_return"].shift(-self.config.direction_window) > 0
        ).astype(int)
        
        # Drop rows with NaN (from lagging and rolling)
        features = features.dropna()
        
        return features
    
    def _get_feature_names(self) -> list:
        """Return list of engineered feature names (excluding targets)."""
        return [
            "log_return",
            *[f"return_lag_{lag}" for lag in self.config.lag_windows],
            *[f"volatility_{w}" for w in self.config.volatility_windows],
            "hl_spread",
            "volume_ma_ratio",
        ]


def build_pipeline(config: Optional[DataConfig] = None) -> PipelineOutput:
    """
    Convenience function to build the pipeline in one call.
    
    Args:
        config: DataConfig object (optional)
        
    Returns:
        PipelineOutput
    """
    pipeline = DataPipeline(config)
    return pipeline.run()
