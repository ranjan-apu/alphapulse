"""
Walk-forward replay clock. Moves through configured intraday candles one step
at a time, ensuring the agent only sees data <= decision time T.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Iterator, Tuple
import pytz

from config import config
from core.context import has_full_micro_context

IST = pytz.timezone("Asia/Kolkata")


def get_session_times(date: datetime) -> Tuple[datetime, datetime, datetime]:
    """
    Get session start, decision start, and session end for a given date.
    All times in IST.
    """
    base = date.replace(hour=0, minute=0, second=0, microsecond=0)

    session_start = base.replace(
        hour=config.SESSION_START_HOUR,
        minute=config.SESSION_START_MINUTE,
    )
    decision_start = base.replace(
        hour=config.DECISION_START_HOUR,
        minute=config.DECISION_START_MINUTE,
    )
    session_end = base.replace(
        hour=config.SESSION_END_HOUR,
        minute=config.SESSION_END_MINUTE,
    )

    return session_start, decision_start, session_end


def is_decision_eligible(ts: datetime, data_intraday: pd.DataFrame) -> bool:
    """
    Check if timestamp ts is eligible for a decision.
    Must be within session and after decision_start.
    Must have at least 1 candle after it for evaluation.
    """
    session_start, decision_start, session_end = get_session_times(ts)

    if ts < decision_start:
        return False
    if ts > session_end:
        return False

    # Need at least one future candle for evaluation
    future = data_intraday[data_intraday.index > ts]
    if len(future) == 0:
        return False

    if config.REQUIRE_FULL_MICRO_CONTEXT and not has_full_micro_context(data_intraday, ts):
        return False

    return True


class WalkForwardClock:
    """
    Iterator that yields one configured intraday decision point at a time.
    At each step T, provides:
    - decision_time: Timestamp of current decision point
    - candle_T: The candle at time T (the closed intraday candle)
    - data_up_to_T: All intraday data up to and including T
    """

    def __init__(self, data_5m: pd.DataFrame):
        self.data_5m = data_5m.sort_index()

    def iterate(self) -> Iterator[dict]:
        """
        Yield decision points. Each yield provides:
        {
            'decision_time': datetime,
            'candle_T': Series (the closed 5m candle at T),
            'data_up_to_T': DataFrame (all 5m data <= T),
            'session_start': datetime,
            'session_end': datetime,
        }
        """
        timestamps = self.data_5m.index.tolist()

        for i, T in enumerate(timestamps):
            if not is_decision_eligible(T, self.data_5m):
                continue

            session_start, _, session_end = get_session_times(T)

            # Slice data up to T (inclusive)
            data_up_to_T = self.data_5m.loc[:T].copy()

            # The candle at T
            candle_T = self.data_5m.loc[T]

            yield {
                "decision_time": T,
                "candle_T": candle_T,
                "data_up_to_T": data_up_to_T,
                "session_start": session_start,
                "session_end": session_end,
            }

    def total_steps(self) -> int:
        """Count total decision steps."""
        count = 0
        for T in self.data_5m.index:
            if is_decision_eligible(T, self.data_5m):
                count += 1
        return count
