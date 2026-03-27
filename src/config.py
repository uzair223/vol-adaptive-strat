"""Configuration models for the vol-adaptive strategy using Pydantic v2."""

from pathlib import Path
import re
from typing import Any, Dict, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator

class HashableBaseModel(BaseModel):
    def __hash__(self) -> int:
        return hash(self.model_dump_json())

class Asset(HashableBaseModel):
    """Asset configuration."""

    yf: str = Field(..., description="Yahoo Finance symbol")
    apca: str = Field(..., description="Alpaca asset symbol")
    name: str = Field(..., description="Human-readable asset name")
    regime_weights: Dict[str, float] = Field(
        ..., description="Regime-specific weights"
    )
    model_config = {"extra": "forbid"}


class TradingConfig(HashableBaseModel):
    """Trading configuration."""
    
    paper_trading: bool = Field(default=True, description="Use paper trading mode")
    allow_fractional_shares: bool = Field(default=True, description="Allow fractional shares for better position sizing")
    assets: list[Asset] = Field(..., description="List of assets to trade")
    market_open: str = Field(default="09:30", description="Market open time (HH:MM)")
    market_close: str = Field(default="16:00", description="Market close time (HH:MM)")
    timezone: str = Field(default="US/Eastern", description="Market timezone")
    rebalance_freq: str = Field(default="1d", description="Rebalance frequency (e.g. '1d', '1w', '1mo')")
    min_rebalance_threshold: float = Field(
        ge=0, le=1, default=0.05, description="Minimum allocation change to trigger rebalance (0-1)"
    )
    
    model_config = {"extra": "forbid"}

    @field_validator("market_open", "market_close", mode="before")
    @classmethod
    def validate_time_format(cls, v):
        """Validate time format is HH:MM."""
        if not isinstance(v, str) or len(v.split(":")) != 2:
            raise ValueError("Time must be in HH:MM format")
        return v

    @field_validator("rebalance_freq", mode="before")
    @classmethod
    def validate_rebalance_freq(cls, v):
        """Validate rebalance frequency format like 1d, 1w, 1m, 1mo, 1q."""
        if not isinstance(v, str):
            raise ValueError("rebalance_freq must be a string")
        if re.fullmatch(r"\d+(d|w|m|mo|q)", v.strip().lower()) is None:
            raise ValueError("rebalance_freq must match '<N><unit>' where unit is d|w|m|mo|q")
        return v.strip().lower()


class DataConfig(HashableBaseModel):
    """Data fetching configuration."""

    lookback_days: int = Field(ge=1, description="Historical data lookback in days")
    feed: Literal["iex", "sip"] = Field(default="iex", description="Data feed source")

    model_config = {"extra": "forbid"}


class StrategyConfig(HashableBaseModel):
    """Strategy parameters."""

    target_vol: float = Field(gt=0, le=1, description="Target annualized volatility (0-1)")
    vol_lookback: int = Field(
        ge=5, le=252, description="Volatility lookback window (trading days)"
    )
    vol_floor: float = Field(ge=0, le=0.1, description="Minimum realized volatility")
    max_exposure: float = Field(ge=1, description="Maximum portfolio leverage")
    base_leverage: float = Field(ge=0.1, description="Base leverage for regime weighting")
    conf_min: float = Field(ge=0, le=1, description="Minimum confidence threshold (0-1)")

    model_config = {"extra": "forbid"}


class RegimeDetectorConfig(HashableBaseModel):
    """Regime detector configuration."""

    n_init: int = Field(ge=1, le=100, description="HMM initialization attempts")
    n_iter: int = Field(ge=100, le=10000, description="HMM training iterations")
    random_state: int = Field(ge=0, description="Random seed for reproducibility")
    retrain_interval_days: int = Field(ge=1, le=365, description="Retraining frequency (days)")
    retrain_window_days: int = Field(ge=20, le=1260, description="Training data window (days)")
    min_holding_period: int = Field(ge=1, le=100, description="Min bars in regime")
    transition_penalty: float = Field(ge=0, le=1, description="Regime transition penalty")
    low_confidence_threshold: float = Field(
        ge=0, le=1, description="Low confidence threshold (0-1)"
    )

    model_config = {"extra": "forbid"}


class Config(HashableBaseModel):
    """Root configuration with validation."""

    trading: TradingConfig
    data: DataConfig
    strategy: StrategyConfig
    regime_detector: RegimeDetectorConfig

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file with validation."""
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump()

    def update_from_dict(self, updates: dict) -> None:
        """
        Update configuration from a dictionary (supports nested keys).
        
        Validates all updates before applying them. Raises ValidationError if invalid.
        
        Parameters
        ----------
        updates : dict
            Dictionary with updates. Supports nested dicts.
            
        Example
        -------
        config.update_from_dict({
            "strategy": {"target_vol": 0.10},
            "data": {"lookback_days": 630}
        })
        """
        # Merge updates into current config
        current_dict = self.model_dump()
        Config._deep_merge(current_dict, updates)
        
        # Validate entire config with merged values
        updated_config = Config.model_validate(current_dict)
        
        # Update self with validated values
        self.trading = updated_config.trading
        self.data = updated_config.data
        self.strategy = updated_config.strategy
        self.regime_detector = updated_config.regime_detector

    @staticmethod
    def get_nested_value(data: Union[dict, "Config"], key_path: str) -> Any:
        """
        Retrieve value from nested config using dot notation.
        
        Parameters
        ----------
        data : dict or Config
            Configuration data (dict or Config object)
        key_path : str
            Dot-separated path to value (e.g., "strategy.target_vol")
            
        Returns
        -------
        Any
            Value at key_path, or None if not found
        """
        # Convert Config object to dict if needed
        if isinstance(data, Config):
            data = data.model_dump()
        
        keys = key_path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    @staticmethod
    def _deep_merge(target: dict, source: dict) -> None:
        """Recursively merge source dict into target dict (modifies target in-place)."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                Config._deep_merge(target[key], value)
            else:
                target[key] = value
