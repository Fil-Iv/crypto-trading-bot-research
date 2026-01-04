# data_cache.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

import pandas as pd

CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/ohlcv"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_MIN = int(os.getenv("CACHE_TTL_MIN", "10"))  # refresh if older than N minutes

class OHLCVCache:
    """Parquet-backed OHLCV cache with append of missing candles and TTL refresh."""
    def __init__(self, base: Path = CACHE_DIR):
        self.base = base

    def _path(self, symbol: str, timeframe: str) -> Path:
        s = symbol.replace("/", "_")
        return self.base / f"{s}_{timeframe}.parquet"

    def _stale(self, p: Path) -> bool:
        if not p.exists():
            return True
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
        return age_min >= CACHE_TTL_MIN

    def read(self, symbol: str, timeframe: str) -> pd.DataFrame:
        p = self._path(symbol, timeframe)
        if not p.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(p)

            # ✅ FIX: правилно обработване на timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
                df.index.name = None
                if "timestamp" in df.columns:
                    del df["timestamp"]
            else:
                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = None

            return df
        except Exception:
            return pd.DataFrame()

    def write(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        p = self._path(symbol, timeframe)
        d = df.copy()
        if "timestamp" not in d.columns:
            d["timestamp"] = d.index
        d = d.reset_index(drop=True).sort_values("timestamp")
        d.to_parquet(p, index=False)

    def merge_save(self, symbol: str, timeframe: str, new_df: pd.DataFrame) -> pd.DataFrame:
        if new_df.empty:
            return new_df
        old = self.read(symbol, timeframe)
        if old.empty:
            self.write(symbol, timeframe, new_df)
            return new_df
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self.write(symbol, timeframe, combined)
        return combined

    def fetch(self, exchange, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        p = self._path(symbol, timeframe)
        stale = self._stale(p)
        if not stale:
            return self.read(symbol, timeframe)

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.index.name = None
            self.write(symbol, timeframe, df)
            return df
        except Exception:
            return self.read(symbol, timeframe)