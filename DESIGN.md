# Data Pipeline Module Design

## Overview
The data pipeline is the foundational layer. It orchestrates ingestion, validation, and feature engineering in a modular, reusable manner.

## Design Principles

### 1. **Separation of Concerns**
- `config.py`: Configuration and constants (single source of truth)
- `validation.py`: Data quality checks (testable, reusable)
- `pipeline.py`: Orchestration and feature engineering
- `example_usage.py`: Demonstration and integration testing

### 2. **No Data Leakage**
- Features are computed forward in time only
- Direction label uses `shift(-forecast_horizon)` to ensure we predict *future* returns
- All time-indexed operations preserve temporal causality
- No look-ahead bias

### 3. **Immutability & Type Safety**
- `PipelineOutput` is a dataclass that freezes returned DataFrames
- Prevents accidental modification by downstream code
- Includes metadata (config, validation report, feature names)

### 4. **Production-Grade Validation**
- `DataValidator.validate_ohlcv()` checks:
  - Required columns and data types
  - Minimum record count
  - Missing value tolerance
  - OHLC logical constraints (High ≥ Open, High ≥ Low, etc.)
  - Volume sanity
  - Return outliers (z-score detection)
  - Date continuity
- Returns detailed `ValidationReport` with warnings
- Fails fast on validation errors

### 5. **Robustness**
- Forward-fills missing values (up to limit)
- Handles timezone normalization (UTC)
- Drops rows with engineered NaN (from lagging/rolling windows)
- Configurable windows and lag structures

## Features Engineered

### Returns
- `log_return`: Daily log return
- `return_lag_{k}`: Lagged log returns (configurable lags)

### Volatility
- `volatility_{w}`: Rolling standard deviation of returns (configurable windows)
- `hl_spread`: Intraday volatility proxy (log of High/Low ratio)

### Volume
- `volume_ma_ratio`: Current volume / 20-day MA (detects volume spikes)

### Target
- `direction`: Binary label = 1 if log_return(t+h) > 0, else 0
  - Forward-shifted by `direction_forward_window` (default: 1 day)
  - Encodes "directional predictability"

## API

### `DataPipeline(config: Optional[DataConfig]) -> PipelineOutput`
```python
pipeline = DataPipeline(config)
output = pipeline.run()

# Access data
df = output.df  # Full feature matrix
features = output.get_features(include_target=False)
X_y = output.get_features(include_target=True)

# Inspect
print(output.summary())
print(output.validation_report)
```

### Convenience Function
```python
output = build_pipeline(config=None)  # Uses defaults if config is None
```

## Configuration

See `config.py` for all tunables:
- `DATA_TICKER`, `DATA_START_DATE`, `DATA_END_DATE`
- `VOLATILITY_WINDOWS`, `LAG_WINDOWS`
- `DIRECTION_FORWARD_WINDOW`
- `MIN_DATA_POINTS`, `MAX_MISSING_PCT`, `OUTLIER_ZSCORE_THRESHOLD`

Customize via `DataConfig`:
```python
config = DataConfig(
    ticker="SPY",
    start_date="2015-01-01",
    volatility_windows=[5, 20],
    lag_windows=[1, 5, 10],
)
output = build_pipeline(config)
```

## Why This Structure?

1. **Reusability**: `DataValidator` and `DataPipeline` are independent; can validate other datasets
2. **Testability**: Each module has a single responsibility; easy to unit test
3. **Production Readiness**: Logging, error handling, immutability, schema validation
4. **Maintainability**: Clear separation between config, logic, and usage
5. **Extensibility**: Easy to add new features without refactoring core pipeline

## Next Steps

Once this is locked in, we'll build:
- **Walk-Forward Validation Engine**: Uses this pipeline output, applies train/test splits
- **Baseline Models**: Accept PipelineOutput, return predictions
- **Regime Detection**: Analyzes the engineered features and raw data
