"""
Stock metadata population from yfinance.
Handles corporate actions (splits, dividends, bonuses),
adjustment factors, and circuit limits.

The plan requires:
- adjusted close for historical context (charting, indicators)
- raw unadjusted prices for trade execution
- Harness must store both and never mix them
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
import json


@dataclass
class StockMetadata:
    """Complete stock reference data and corporate actions."""
    symbol: str
    isin: Optional[str] = None
    lot_size: int = 1
    circuit_limit_upper: Optional[float] = None
    circuit_limit_lower: Optional[float] = None
    adjustment_factor: float = 1.0  # Multiply unadjusted by this to get adjusted
    yahoo_ticker: Optional[str] = None
    expiry_cycle: Optional[str] = None  # null for equity_cash
    is_index: bool = False

    # Corporate actions
    earnings_dates: List[dict] = field(default_factory=list)
    split_dates: List[dict] = field(default_factory=list)
    dividend_dates: List[dict] = field(default_factory=list)
    bonus_dates: List[dict] = field(default_factory=list)

    notes: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


class StockMetadataManager:
    """
    Populates and manages StockMetadata from various sources.

    Primary source: yfinance ticker info.
    Provides adjustment factors for split/dividend-adjusted prices.
    """

    def __init__(self):
        self._metadata: Dict[str, StockMetadata] = {}

    def populate_from_yfinance(self, symbol: str, yahoo_ticker: str) -> StockMetadata:
        """
        Populate stock metadata from yfinance.

        Args:
            symbol: Internal symbol (e.g., "RELIANCE")
            yahoo_ticker: Yahoo Finance ticker (e.g., "RELIANCE.NS")

        Returns:
            StockMetadata with populated fields
        """
        metadata = StockMetadata(
            symbol=symbol,
            yahoo_ticker=yahoo_ticker,
            is_index=False,
        )

        try:
            import yfinance as yf
            ticker = yf.Ticker(yahoo_ticker)
            info = ticker.info or {}

            # Basic info
            metadata.isin = info.get("isin")
            metadata.lot_size = info.get("lotSize", 1)

            # Circuit limits (price bands)
            metadata.circuit_limit_upper = info.get("priceBandUpper") or info.get("dayHigh")
            metadata.circuit_limit_lower = info.get("priceBandLower") or info.get("dayLow")

            # Splits
            try:
                splits = ticker.splits
                if splits is not None and len(splits) > 0:
                    for idx, ratio in splits.items():
                        metadata.split_dates.append({
                            "date": str(idx.date()),
                            "ratio": float(ratio),
                        })
            except Exception:
                pass

            # Dividends
            try:
                dividends = ticker.dividends
                if dividends is not None and len(dividends) > 0:
                    for idx, amount in dividends.items():
                        if float(amount) > 0:
                            metadata.dividend_dates.append({
                                "date": str(idx.date()),
                                "amount": float(amount),
                                "type": "historical",
                            })
            except Exception:
                pass

            # Compute adjustment factor from splits
            metadata.adjustment_factor = self._compute_adjustment_factor(
                metadata.split_dates
            )

            metadata.updated_at = datetime.now().astimezone()
            metadata.notes = f"Populated from yfinance ({yahoo_ticker})"

        except ImportError:
            metadata.notes = "yfinance not available; metadata populated with defaults"
        except Exception as e:
            metadata.notes = f"Error populating from yfinance: {e}"

        self._metadata[symbol] = metadata
        return metadata

    @staticmethod
    def _compute_adjustment_factor(split_dates: List[dict]) -> float:
        """
        Compute cumulative adjustment factor from split history.

        A 2:1 split means the adjustment factor is 2.0
        (multiply old prices by 2 to get split-adjusted).
        """
        factor = 1.0
        for split in split_dates:
            ratio = split.get("ratio", 1.0)
            if ratio > 0:
                factor *= ratio
        return factor

    def get_adjustment_factor(self, symbol: str) -> float:
        """Get the price adjustment factor for a symbol."""
        meta = self._metadata.get(symbol)
        return meta.adjustment_factor if meta else 1.0

    def adjust_prices(self, symbol: str, df_unadjusted, reverse: bool = False):
        """
        Adjust historical prices for splits.

        Args:
            symbol: Stock symbol
            df_unadjusted: DataFrame with unadjusted prices
            reverse: If True, convert adjusted back to unadjusted

        Returns:
            DataFrame with adjusted OHLCV columns
        """
        factor = self.get_adjustment_factor(symbol)
        if factor == 1.0:
            return df_unadjusted.copy()

        df = df_unadjusted.copy()
        if reverse:
            # Adjusted -> Unadjusted
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] / factor
        else:
            # Unadjusted -> Adjusted
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * factor

        return df

    def get_corporate_action_dates(
        self,
        symbol: str,
        action_type: str,
        start_date: date,
        end_date: date,
    ) -> List[dict]:
        """
        Get corporate actions within a date range.

        Args:
            symbol: Stock symbol
            action_type: 'split', 'dividend', 'bonus', or 'earnings'
            start_date: Start of range
            end_date: End of range

        Returns:
            List of action dicts within the range
        """
        meta = self._metadata.get(symbol)
        if not meta:
            return []

        actions = {
            "split": meta.split_dates,
            "dividend": meta.dividend_dates,
            "bonus": meta.bonus_dates,
            "earnings": meta.earnings_dates,
        }.get(action_type, [])

        result = []
        for action in actions:
            action_date_str = action.get("date", "")
            try:
                action_date = date.fromisoformat(action_date_str)
                if start_date <= action_date <= end_date:
                    result.append(action)
            except (ValueError, TypeError):
                continue

        return result

    def has_earnings_nearby(
        self,
        symbol: str,
        decision_date: date,
        days_window: int = 5,
    ) -> bool:
        """
        Check if earnings are within N days of a decision date.

        Important for the context builder to flag earnings-driven moves.
        """
        from datetime import timedelta

        meta = self._metadata.get(symbol)
        if not meta:
            return False

        for earnings in meta.earnings_dates:
            try:
                ed = date.fromisoformat(earnings.get("date", ""))
                if abs((decision_date - ed).days) <= days_window:
                    return True
            except (ValueError, TypeError):
                continue

        return False

    def get_metadata(self, symbol: str) -> Optional[StockMetadata]:
        """Get stored metadata for a symbol."""
        return self._metadata.get(symbol)

    def to_dict(self, symbol: str) -> dict:
        """Serialize metadata for Postgres storage."""
        meta = self._metadata.get(symbol)
        if not meta:
            return {}

        return {
            "symbol": meta.symbol,
            "isin": meta.isin,
            "lot_size": meta.lot_size,
            "circuit_limit_upper": meta.circuit_limit_upper,
            "circuit_limit_lower": meta.circuit_limit_lower,
            "adjustment_factor": meta.adjustment_factor,
            "yahoo_ticker": meta.yahoo_ticker,
            "expiry_cycle": meta.expiry_cycle,
            "is_index": meta.is_index,
            "earnings_dates": json.dumps(meta.earnings_dates),
            "split_dates": json.dumps(meta.split_dates),
            "dividend_dates": json.dumps(meta.dividend_dates),
            "bonus_dates": json.dumps(meta.bonus_dates),
            "notes": meta.notes,
            "updated_at": meta.updated_at.isoformat(),
        }
