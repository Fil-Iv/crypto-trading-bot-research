# utils/get_top_pairs.py


from __future__ import annotations
from typing import List, Optional, Set
from functools import lru_cache
import os
# This module expects to live under a package (e.g. utils). Import logger relatively.
try:
    from .logger import log
except Exception:  # allow running as script
    def log(msg: str) -> None: print(msg)

_cached_symbols: List[str] = []

def _usd_volume_of_ticker(symbol: str, t: dict) -> float:
    """Best-effort USD volume from a ccxt-like ticker dict."""
    if not isinstance(t, dict):
        return 0.0
    # Prefer quoteVolume if quote is USDC
    quote_vol = t.get('quoteVolume') or t.get('info', {}).get('quoteVolume')
    if quote_vol is not None and symbol.endswith('/USDC'):
        try:
            return float(quote_vol)
        except Exception:
            pass
    # Else try baseVolume * last
    base_vol = t.get('baseVolume') or t.get('info', {}).get('baseVolume')
    last = t.get('last') or t.get('close') or t.get('info', {}).get('last')
    try:
        return float(base_vol) * float(last)
    except Exception:
        return 0.0

def get_top_usdc_pairs(exchange,
                       limit: int = 12,
                       min_volume_usd: float = 1_000_000.0,
                       blacklist: Optional[Set[str]] = None) -> List[str]:
    """
    Return top /USDC symbols by estimated USD volume.
    Falls back to a fixed list if tickers are unavailable.
    Results are cached in-process for reuse.
    """
    global _cached_symbols
    blacklist = blacklist or set()
    symbols: List[str] = []
    # прочети blacklist от .env, напр. SYMBOLS_BLACKLIST=FDUSD/USDC,WLFI/USDC
    env_bl = (os.getenv("SYMBOLS_BLACKLIST", "") or "").strip()
    extra_blacklist = set(x.strip() for x in env_bl.split(",") if x.strip())

    try:
        # Try tickers first (fastest for volumes)
        tickers = exchange.fetch_tickers()
        candidates = [
            s for s in tickers.keys()
            if s.endswith('/USDC') and s not in blacklist and s not in extra_blacklist
        ]

        scored = [(s, _usd_volume_of_ticker(s, tickers.get(s) or {})) for s in candidates]
        # filter min volume
        scored = [x for x in scored if x[1] >= min_volume_usd]
        # sort desc by volume
        symbols = [s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        log(f"[pairs] fetched from tickers: {len(symbols)} symbols")
    except Exception as e:
        log(f"[pairs] fetch_tickers failed: {e}")

    if not symbols:
        # Fallback to markets if tickers missing
        try:
            markets = exchange.load_markets()
            candidates = [m for m in markets.values() if m.get('quote') == 'USDC']
            # Prefer by 'info' or 'limits' if volume missing; no reliable volume -> alphabetical but filtered
            symbols = sorted({
                m['symbol'] for m in candidates
                if m['symbol'] not in blacklist and m['symbol'] not in extra_blacklist
            })

            log(f"[pairs] loaded from markets: {len(symbols)} symbols")
        except Exception as e:
            log(f"[pairs] load_markets failed: {e}")

    if not symbols:
        # Final fixed fallback
        symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC", "ADA/USDC", "AVAX/USDC"]

    _cached_symbols = symbols[:limit]
    return _cached_symbols
