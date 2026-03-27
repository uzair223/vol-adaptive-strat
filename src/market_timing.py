"""
market_timing.py
================
Market timing utilities for trading schedule management.

Encapsulates all market timing logic including:
- Market open/close time parsing
- Trading day identification (Mon-Fri)
- Market hours checking
- Timezone handling
"""

from datetime import datetime, time
from typing import Callable
import pytz

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import Config


class MarketTiming:
    """
    Manages market timing logic for trading schedule.
    
    Handles market open/close time parsing, trading day identification,
    and time comparisons in the configured market timezone.
    """
    
    def __init__(self, config: Config):
        """
        Initialize market timing with configuration.
        
        Parameters
        ----------
        config : Config
            Configuration object containing trading.market_open, 
            trading.market_close, and trading.timezone
        """
        self.config = config
        self._timezone = pytz.timezone(config.trading.timezone)
        
        # Parse market open and close times once during initialization
        open_hour, open_min = map(int, config.trading.market_open.split(":"))
        close_hour, close_min = map(int, config.trading.market_close.split(":"))
        
        self._market_open_time = time(open_hour, open_min)
        self._market_close_time = time(close_hour, close_min)
    
    @property
    def market_open_time(self) -> time:
        """Return parsed market open time as time object."""
        return self._market_open_time
    
    @property
    def market_close_time(self) -> time:
        """Return parsed market close time as time object."""
        return self._market_close_time
    
    @property
    def timezone(self) -> str:
        """Return market timezone string."""
        return self.config.trading.timezone
    
    def is_trading_day(self, dt: datetime | None = None) -> bool:
        """
        Check if given datetime is on a trading day (Monday-Friday).
        
        Parameters
        ----------
        dt : datetime | None
            Datetime to check. If None, uses current time in market timezone.
        
        Returns
        -------
        bool
            True if Monday-Friday (weekday < 5), False otherwise.
        """
        if dt is None:
            dt = self.get_current_time_in_market_tz()
        return dt.weekday() < 5
    
    def is_market_open(self, dt: datetime | None = None) -> bool:
        """
        Check if market is currently open at given time.
        
        Returns True if between market_open and market_close on a trading day.
        
        Parameters
        ----------
        dt : datetime | None
            Datetime to check. If None, uses current time in market timezone.
        
        Returns
        -------
        bool
            True if market is open, False otherwise.
        """
        if dt is None:
            dt = self.get_current_time_in_market_tz()
        
        # Ensure datetime is in market timezone
        if dt.tzinfo is None:
            dt = self._timezone.localize(dt)
        elif dt.tzinfo != self._timezone:
            dt = dt.astimezone(self._timezone)
        
        # Check if trading day and within market hours
        is_trading_day = self.is_trading_day(dt)
        now_time = dt.time()
        is_market_hours = self._market_open_time <= now_time < self._market_close_time
        
        return is_trading_day and is_market_hours
    
    def should_fire_at_startup(self) -> bool:
        """
        Determine if market-open event should fire immediately at startup.
        
        Returns True if market is currently open, indicating startup during
        trading hours when previous day's data should be processed.
        
        Returns
        -------
        bool
            True if market is currently open, False otherwise.
        """
        return self.is_market_open()
    
    def get_current_time_in_market_tz(self) -> datetime:
        """
        Get current time in market timezone.
        
        Returns
        -------
        datetime
            Current datetime in market timezone with tzinfo set.
        """
        return datetime.now(self._timezone)
    
    def add_market_open_job(
        self, 
        scheduler: BackgroundScheduler, 
        callback: Callable
    ) -> None:
        """
        Add market open scheduled job to the provided scheduler.
        
        Parameters
        ----------
        scheduler : BackgroundScheduler
            APScheduler scheduler to add job to
        callback : Callable
            Callback function to execute at market open
        """
        scheduler.add_job(
            callback,
            CronTrigger(
                hour=self.market_open_time.hour,
                minute=self.market_open_time.minute,
                day_of_week="0-4"
            ),
            timezone=self.timezone,
            id="market_open"
        )
