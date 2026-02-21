"""
Configuration and constants for the regime-aware forecasting project.
"""

from dataclasses import dataclass
from typing import Literal

# Data retrieval
DATA_TICKER = "SPY"
DATA_START_DATE = "2010-01-01"
DATA_END_DATE = "2025-01-31"
DATA_FREQUENCY = "1d"

# Feature engineering
VOLATILITY_WINDOWS = [5, 20, 60]  # Trading days
LAG_WINDOWS = [1, 2, 5, 10, 20]  # Days
DIRECTION_FORWARD_WINDOW = 1  # Days ahead for direction label

# Validation
MIN_DATA_POINTS = 252  # ~1 year of trading days
MAX_MISSING_PCT = 0.05  # Allow up to 5% missing values
OUTLIER_ZSCORE_THRESHOLD = 5.0  # Z-score for outlier detection

# Walk-forward validation defaults
DEFAULT_TRAIN_SIZE = 252 * 2  # 2 years
DEFAULT_TEST_SIZE = 252 // 12  # 1 month
DEFAULT_FORECAST_HORIZON = 1  # Days

# Regime detection
VOLATILITY_PERCENTILE_BREAKS = [0.33, 0.67]  # Low/medium/high regimes


@dataclass
class DataConfig:
    """Configuration container for pipeline runs."""
    ticker: str = DATA_TICKER
    start_date: str = DATA_START_DATE
    end_date: str = DATA_END_DATE
    volatility_windows: list = None
    lag_windows: list = None
    direction_window: int = DIRECTION_FORWARD_WINDOW
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = VOLATILITY_WINDOWS
        if self.lag_windows is None:
            self.lag_windows = LAG_WINDOWS
