# utils/capital_manager.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone, date

# -------- logger fallback --------
try:
    from .logger import log  # when inside utils package
except Exception:
    try:
        from logger import log  # top-level
    except Exception:
        def log(x: str) -> None: print(x)

# -------- optional storage (for max positions) --------
try:
    from .storage import get_open_positions  # utils.storage
except Exception:
    try:
        from storage import get_open_positions  # top-level
    except Exception:
        def get_open_positions() -> list[dict[str, Any]]:
            return []

# -------- persistent state --------
STATE_PATH = Path(os.getenv("CAPITAL_STATE_PATH", "data/capital_state.json"))
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _today_str() -> str:
    return date.today().isoformat()

def _load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"available_quote": 0.0, "open_alloc_quote": 0.0, "daily_pnl": {}}

def _save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"[capital] save state error: {e}")

# -------- config --------
@dataclass
class RiskConfig:
    quote_ccy: str = os.getenv("QUOTE_CCY", "USDC")
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.10"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
    max_daily_loss_quote: float = float(os.getenv("MAX_DAILY_LOSS_QUOTE", "200.0"))
    max_trade_quote: float = float(os.getenv("MAX_TRADE_QUOTE", "10000.0"))
    min_quote_alloc: float = float(os.getenv("MIN_QUOTE_ALLOC", "10.0"))
    max_exposure_pct: float = float(os.getenv("MAX_EXPOSURE_PCT", "0.30"))
    trade_alloc_pct: float = float(os.getenv("TRADE_ALLOC_PCT", "0.90"))
    aggressiveness: float = float(os.getenv("AGGRESSIVENESS", "1.0"))
    notional_buffer: float = float(os.getenv("NOTIONAL_BUFFER", "1.2"))  # 20% headroom

# -------- helpers --------
def _safe_float(x: Any, fb: float = 0.0) -> float:
    try:
        if isinstance(x, dict):
            for k in ("price", "last", "close", "average", "avgPrice"):
                if k in x:
                    x = x[k]
                    break
        return float(x)
    except Exception:
        return float(fb)

def _price_from_ticker(t: dict, fallback: float = 0.0) -> float:
    cands = [
        t.get("average"), t.get("price"), t.get("last"), t.get("close"),
        (t.get("info") or {}).get("price"),
        (t.get("info") or {}).get("last"),
        (t.get("info") or {}).get("close"),
    ]
    for v in cands:
        if v is None:
            continue
        if isinstance(v, dict):
            for k in ("price", "last", "close", "average", "avgPrice"):
                if k in v:
                    v = v[k]
                    break
        try:
            return float(v)
        except Exception:
            continue
    return float(fallback)

def _round_to_step(value: float, step: float) -> float:
    if step is None or step <= 0:
        return value
    k = int(value / step)
    return k * step

# -------- manager --------
class CapitalManager:
    def __init__(self):
        self.cfg = RiskConfig()
        self.state = _load_state()

    # -- balance sync --
    def fetch_available_quote(self, exchange) -> float:
        try:
            bal = exchange.fetch_balance() or {}
            free = bal.get("free") or bal.get("total") or {}
            amt = float(free.get(self.cfg.quote_ccy, 0.0))
            self.state["available_quote"] = amt
            _save_state(self.state)
            return amt
        except Exception as e:
            log(f"[capital] fetch_balance error: {e}")
            return float(self.state.get("available_quote", 0.0))

    def get_available_capital(self, exchange=None) -> float:
        if exchange is not None:
            return self.fetch_available_quote(exchange)
        return float(self.state.get("available_quote", 0.0))

    # -- allocation accounting --
    def update_after_trade(self, symbol: str, quote_delta: float, pnl_quote: float = 0.0) -> None:
        """
        Update the internal capital state after a trade.

        The ``quote_delta`` parameter reflects how much quote currency was
        allocated or freed for a position.  A positive ``quote_delta`` is
        interpreted as capital being locked into a new trade, whereas a
        negative value indicates the position has been closed and capital is
        returned.  ``pnl_quote`` captures the realised profit or loss from
        the trade in quote currency.

        This method updates three pieces of state:

        * ``open_alloc_quote`` — the total amount of quote currency currently
          allocated to open positions.  It increases when a new trade is
          opened and decreases when a position is closed.
        * ``available_quote`` — our notion of liquid quote capital that can
          be used for new trades.  When opening a trade we reduce the
          available capital by the amount allocated; when closing we return
          the freed capital and add/subtract the realised PnL.  This allows
          the bot to compound gains (or absorb losses) over time when
          running in a simulated or non‑exchange environment where balances
          aren't updated automatically.
        * ``daily_pnl`` — a running record of realised PnL per day.  Only
          updated when ``pnl_quote`` is non‑zero.

        Parameters
        ----------
        symbol: str
            The trading symbol (unused here but kept for future hooks).
        quote_delta: float
            Positive for capital allocated to a new position, negative when
            capital is freed upon closing.  The magnitude should equal the
            notional of the position at entry.
        pnl_quote: float, default 0.0
            Realised profit or loss when closing the position.  Positive for
            profits and negative for losses.
        """
        # Ensure keys exist
        self.state.setdefault("open_alloc_quote", 0.0)
        self.state.setdefault("available_quote", 0.0)
        # Track allocation of quote currency to open positions
        self.state["open_alloc_quote"] = max(
            0.0, float(self.state["open_alloc_quote"]) + float(quote_delta)
        )
        # Adjust available capital: subtract allocations, add back when freed and
        # incorporate realised PnL.  For an open trade (positive quote_delta),
        # this reduces available capital; for a closed trade (negative
        # quote_delta) it increases available capital by the original notional
        # plus any realised PnL.
        try:
            avail = float(self.state.get("available_quote", 0.0))
            # Remove allocated capital and add realised PnL when provided
            avail = avail - float(quote_delta) + float(pnl_quote)
            # Do not allow negative available capital
            if avail < 0.0:
                avail = 0.0
            self.state["available_quote"] = avail
        except Exception:
            pass
        # Update daily PnL record
        if pnl_quote:
            today = _today_str()
            self.state.setdefault("daily_pnl", {})
            self.state["daily_pnl"][today] = float(self.state["daily_pnl"].get(today, 0.0)) + float(pnl_quote)
        _save_state(self.state)

    # -- sizing (live) --
    def size_quote(
        self,
        exchange=None,
        symbol: str = "",
        probability: float = 0.55,
        timeframe: str = "15m",
        aggressiveness: Optional[float] = None,
        tp: float = 0.02,
        sl: float = 0.01,
    ) -> float:
        """
        Връща QUOTE сума за нова сделка, съобразена с:
        - дял от наличния капитал (TRADE_ALLOC_PCT * aggressiveness)
        - твърд минимум от .env (MIN_QUOTE_ALLOC)
        - борсови лимити (minNotional + buffer, minQty/stepSize, precision)
        - общо изложение (MAX_EXPOSURE_PCT)
        - лимит на брой едновременни позиции (MAX_POSITIONS)
        """
        # Aggressiveness multiplier (default to config value if not provided).
        aggr = float(aggressiveness if aggressiveness is not None else self.cfg.aggressiveness)

        # Respect maximum number of concurrent positions; if exceeded, allocate zero.
        try:
            open_pos = get_open_positions()
            if isinstance(open_pos, list) and len(open_pos) >= int(self.cfg.max_positions):
                return 0.0
        except Exception:
            pass

        # Determine available equity in quote currency; if none, allocate zero.
        if exchange is None:
            try:
                from .exchange_factory import get_exchange  # utils
            except Exception:
                from exchange_factory import get_exchange   # top-level
            exchange = get_exchange()
        equity = self.fetch_available_quote(exchange)
        if equity <= 0:
            return 0.0

        # Compute remaining allowable exposure in quote terms.
        cap_total = equity * self.cfg.max_exposure_pct
        used = float(self.state.get("open_alloc_quote", 0.0))
        cap_remaining = max(0.0, cap_total - used)

        min_alloc = float(self.cfg.min_quote_alloc)
        # Ако оставащият експозиционен лимит е под минималния размер на сделка,
        # не отваряме нова позиция. Старото поведение ("форсирай min_alloc")
        # нарушава MAX_EXPOSURE_PCT и води до churn / много дребни сделки.
        if cap_remaining < min_alloc:
            return 0.0


        # Base allocation based on trade allocation percentage and aggressiveness.
        base_alloc = equity * self.cfg.trade_alloc_pct * aggr
        # Cap the allocation by remaining exposure and maximum single-trade limit.
        alloc = min(base_alloc, self.cfg.max_trade_quote, cap_remaining)
        # IMPORTANT: не форсираме минимум тук. Ако Kelly/edge намали allocation-а
        # под min_alloc, това трябва да доведе до "няма сделка", а не до принудителен min.

        # --- Kelly criterion scaling ---
        # To optimise growth while controlling risk, optionally apply a fractional
        # Kelly criterion.  The Kelly formula f = (b*p - q) / b gives the
        # fraction of capital to allocate, where b is the reward‑to‑risk ratio,
        # p is the probability of winning and q = 1 - p.  Full Kelly often
        # produces large bet sizes; fractional Kelly is commonly used to reduce
        # volatility【697492442707852†L142-L179】.  If KELLY_FRACTION > 0, we
        # compute f_k and scale the allocation proportionally.  We use the
        # provided tp/sl to derive b = tp/sl when sl > 0.  A negative
        # f_k indicates an unfavourable expectancy and results in zero allocation.
        kelly_frac = float(self.cfg.kelly_fraction)
        if kelly_frac > 0:
            try:
                b_ratio = float(tp) / float(sl) if sl and float(sl) > 0 else 1.0
                p_win = float(probability)
                # clamp p_win to [0,1]
                if p_win < 0.0:
                    p_win = 0.0
                if p_win > 1.0:
                    p_win = 1.0
                q_loss = 1.0 - p_win
                f_k = (b_ratio * p_win - q_loss) / b_ratio
                if f_k < 0.0:
                    f_k = 0.0
                # Apply fractional Kelly scaling
                f_k *= kelly_frac
                # Multiply allocation by f_k to adjust size.  If f_k is very small,
                # the allocation will be reduced accordingly; if f_k > 1,
                # cap at 1 to avoid over‑sizing.
                if f_k > 1.0:
                    f_k = 1.0
                alloc *= f_k
            except Exception:
                # On any error, skip Kelly scaling
                pass

        # Final safeguard: if allocation is somehow above remaining exposure, trim it.
        if alloc > cap_remaining:
            alloc = cap_remaining

        # Ако след всички скейлинги allocation-а е под минималния, пропускаме.
        if alloc < min_alloc:
            return 0.0

        return float(alloc)

    # -- diagnostics --
    def load_capital_state(self) -> dict:
        return dict(self.state)
