from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone

# ---------- fallbacks for logger/telegram/trade log ----------
try:
    from .logger import log
except Exception:
    def log(msg: str) -> None: print(msg)

try:
    from .notifier import send_telegram
except Exception:
    def send_telegram(msg: str) -> None: pass

try:
    from .trade_logger import log_trade
except Exception:
    def log_trade(*args, **kwargs) -> None: pass

# ---------- fallbacks for storage ----------
try:
    from .storage import (
        insert_order, insert_fill, upsert_open_position,
        close_position, get_open_positions
    )
except Exception:
    def insert_order(*a, **k): pass
    def insert_fill(*a, **k): pass
    def upsert_open_position(*a, **k) -> int: return 0
    def close_position(*a, **k): pass
    def get_open_positions() -> List[Dict[str, Any]]: return []

@dataclass
class Bracket:
    position_id: int
    symbol: str
    side: str
    amount: float
    entry_price: Any       # may be a number or a dict
    tp_price: Optional[float]
    sl_price: Optional[float]
    trail_pct: Optional[float]
    highest: float
    tp1_done: bool = False
    partial_ratio: float = 0.5
    tp1_pct: Optional[float] = None

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _safe_float(x: Any, fb: float = 0.0) -> float:
    try:
        # If x is a dict, pick common numeric keys
        if isinstance(x, dict):
            for key in ("price", "last", "close", "average", "avgPrice"):
                if key in x:
                    x = x[key]
                    break
        return float(x)
    except Exception:
        return float(fb)

def _safe_price_from_ticker(t: dict, fallback: float = 0.0) -> float:
    candidates = [
        t.get('average'),
        t.get('price'),
        t.get('last'),
        t.get('close'),
        (t.get('info') or {}).get('price'),
        (t.get('info') or {}).get('last'),
        (t.get('info') or {}).get('close'),
    ]
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, dict):
            for k in ('price','last','close'):
                if k in v:
                    v = v[k]
                    break
        try:
            return float(v)
        except Exception:
            continue
    return _safe_float(fallback, 0.0)

class TradeManager:
    """
    Responsible for opening/closing long positions and syncing with Binance.
    Keeps an in‑memory dictionary of open Brackets (self.brackets).
    """
    def __init__(self) -> None:
        self.brackets: Dict[str, Bracket] = {}
        self._restore_open_positions()

    def _restore_open_positions(self) -> None:
        try:
            rows = get_open_positions()
            for r in rows:
                entry_px = _safe_float(r.get('entry_price', 0.0), 0.0)
                b = Bracket(
                    position_id=r['id'],
                    symbol=r['symbol'],
                    side=r['side'],
                    amount=float(r['qty_open']),
                    entry_price=entry_px,
                    tp_price=None if r['tp'] is None else float(r['tp']),
                    sl_price=None if r['sl'] is None else float(r['sl']),
                    trail_pct=None if r['trailing'] is None else float(r['trailing']),
                    highest=entry_px
                )
                self.brackets[b.symbol] = b
                log(f"[restore] restored open position {b.symbol} id={b.position_id}")
        except Exception as e:
            log(f"[restore] failed: {e}")

    def sync_manual_positions(self, exchange) -> None:
        """
        Detects any positions opened manually on Binance that are not in self.brackets,
        and registers them. Uses the free balance for each base asset.
        """
        try:
            balances = exchange.fetch_balance() or {}
            prices   = exchange.fetch_tickers()
            for symbol, t in prices.items():
                # we only trade USDC pairs
                if not symbol.endswith("/USDC"):
                    continue
                base = symbol.split("/")[0]
                free = float((balances.get("free") or {}).get(base, 0.0))
                price = _safe_price_from_ticker(t, fallback=0.0)

                # if we hold any base asset free and there's no bracket for it, register as manual long
                if free > 0.0 and price > 0.0 and symbol not in self.brackets:
                    b = Bracket(
                        position_id=-1,
                        symbol=symbol,
                        side='long',
                        amount=free,
                        entry_price=price,
                        tp_price=None,
                        sl_price=None,
                        trail_pct=None,
                        highest=price,
                    )
                    self.brackets[symbol] = b
                    log(f"[manual] Detected manual position: {symbol} x {free:.6f} @ {price:.4f}")
                    try:
                        send_telegram(f"📥 Manual position: {symbol} x {free:.6f} @ {price:.4f}")
                    except Exception:
                        pass
        except Exception as e:
            log(f"[manual sync error] {e}")

    def open_long(
        self,
        exchange,
        symbol: str,
        quote_capital: float,
        tp_pct: Optional[float],
        sl_pct: Optional[float],
        trail_pct: Optional[float],
        partial_ratio: float = 0.5,
        tp1_pct: Optional[float] = None,
    ) -> Optional[Bracket]:
        """
        Executes a market buy order with quote_capital (USDC) if notional >= min_notional.
        Returns a Bracket on success or None otherwise.
        """
        try:
            ticker = exchange.fetch_ticker(symbol)
            price  = _safe_price_from_ticker(ticker, fallback=0.0)
            m      = exchange.markets.get(symbol, {}) if hasattr(exchange, "markets") else {}
            limits = m.get("limits", {}) or {}
            prec   = m.get("precision", {}) or {}
            min_notional = _safe_float((limits.get("cost") or {}).get("min"), 5.0)
            min_amount   = _safe_float((limits.get("amount") or {}).get("min"), 0.0)
            amt_step     = _safe_float((limits.get("amount") or {}).get("step"), 0.0)
            px_prec      = int(prec.get("price") or 8)
            amt_prec     = int(prec.get("amount") or 8)
        except Exception as e:
            log(f"[order] error fetching price/limits for {symbol}: {e}")
            send_telegram(f"⚠️ Cannot get price/limits for {symbol}.")
            return None

        if price <= 0:
            log(f"[order] invalid price for {symbol}: {price}")
            return None

        # compute initial amount & notional
        amount   = quote_capital / price
        notional = amount * price

        # adjust amount by step/precision
        if amt_step and amt_step > 0:
            k = int(amount / amt_step)
            amount = max(min_amount, k * amt_step)
        else:
            amount = max(min_amount, float(f"{amount:.{amt_prec}f}"))
        notional = amount * price

        # ensure notional meets min_notional; enlarge if possible within tolerance
        if notional < min_notional:
            req_amt = min_notional / price
            if amt_step and amt_step > 0:
                import math as _math
                req_k = int(_math.ceil(req_amt / amt_step))
                req_amt = max(min_amount, req_k * amt_step)
            else:
                req_amt = max(min_amount, float(f"{req_amt:.{amt_prec}f}"))
            req_notional = req_amt * price
            if req_notional <= quote_capital * 1.05:
                amount = req_amt
                notional = req_notional
            else:
                log(f"[order] SKIPPED {symbol}: notional {notional:.2f} < min_notional {min_notional:.2f} and required {req_notional:.2f} > capital {quote_capital:.2f}")
                return None

        if notional > quote_capital * 1.05:
            log(f"[order] SKIPPED {symbol}: required notional {notional:.2f} > capital {quote_capital:.2f}")
            return None

        # place the order
        try:
            order = None
            if hasattr(exchange, "create_order"):
                order = exchange.create_order(symbol=symbol, type="market", side="buy", amount=amount)
            elif hasattr(exchange, "create_market_buy_order"):
                order = exchange.create_market_buy_order(symbol, amount)
            else:
                raise AttributeError("Exchange does not support market buy orders")
            fill_px = _safe_price_from_ticker(order, fallback=price)
        except Exception as e:
            log(f"[order] error placing market order: {e}")
            send_telegram(f"🚨 Order failed {symbol}: {e}")
            return None

        # log to storage
        order_info = order or {'type':'market','side':'buy','amount':amount,'price':fill_px}
        insert_order(symbol, 'buy', amount, fill_px, order_info)

        pid = upsert_open_position(
            symbol=symbol,
            side='long',
            qty_open=amount,
            entry_price=fill_px,
            tp=fill_px * (1 + tp_pct) if tp_pct else None,
            sl=fill_px * (1 - sl_pct) if sl_pct else None,
            trailing=trail_pct,
        )

        b = Bracket(
            position_id=pid,
            symbol=symbol,
            side='long',
            amount=amount,
            entry_price=fill_px,
            tp_price=fill_px * (1 + tp_pct) if tp_pct else None,
            sl_price=fill_px * (1 - sl_pct) if sl_pct else None,
            trail_pct=trail_pct,
            highest=fill_px,
            partial_ratio=partial_ratio,
            tp1_pct=tp1_pct,
        )
        self.brackets[symbol] = b

        # update capital exposure
        try:
            from .capital_manager import CapitalManager
        except Exception:
            from capital_manager import CapitalManager
        try:
            cm = CapitalManager()
            cm.update_after_trade(symbol, quote_delta=notional, pnl_quote=0.0)
        except Exception:
            pass

        log_trade(symbol, {"side":"buy","amount":amount,"price":fill_px})
        try:
            send_telegram(f"✅ Opened LONG {symbol} @ {fill_px:.{px_prec}f} for ~{notional:.2f} USDC")
        except Exception:
            pass
        return b

    def _close_long(self, exchange, b: Bracket, reason: str) -> None:
        """
        Sells as much as we truly hold free for the base asset, subject to notional/step/precision.
        Leaves a tiny remainder to avoid dust‐selling and reuses entry price for PnL.
        """
        base, _ = b.symbol.split('/')

        # fetch free balance for base asset
        try:
            bal = exchange.fetch_balance() or {}
            free_bal = float((bal.get('free') or {}).get(base, 0.0))
        except Exception as e:
            log(f"[BALANCE error] {b.symbol}: {e}")
            free_bal = 0.0

        # fetch step & precision
        try:
            m    = exchange.markets.get(b.symbol, {}) if hasattr(exchange,'markets') else {}
            limits = m.get('limits', {}) or {}
            prec  = m.get('precision', {}) or {}
            amt_step = float((limits.get('amount') or {}).get('step') or 0.0)
            amt_prec = int(prec.get('amount') or 8)
        except Exception:
            amt_step = 0.0
            amt_prec = 8

        # leave a small cushion (0.5 %) to avoid insufficient balance errors
        fudge = max(amt_step if amt_step > 0 else 0.0, free_bal * 0.005)
        raw_close = max(0.0, free_bal - fudge)
        amount_to_close = min(b.amount, raw_close)

        if amount_to_close <= 0.0:
            log(f"[LOGIC CLOSE] {b.symbol}: amount_to_close={amount_to_close:.8f}")
            close_position(b.position_id, pnl_quote=0.0)
            try:
                from .capital_manager import CapitalManager
            except Exception:
                from capital_manager import CapitalManager
            try:
                cm = CapitalManager()
                cm.update_after_trade(b.symbol, quote_delta=-_safe_float(b.entry_price,0.0)*float(b.amount), pnl_quote=0.0)
            except Exception:
                pass
            try:
                send_telegram(f"ℹ️ {b.symbol}: no free balance, position closed logically.")
            except Exception:
                pass
            self.brackets.pop(b.symbol, None)
            return

        # round to step/precision
        try:
            if amt_step and amt_step > 0:
                k = int(amount_to_close / amt_step)
                amount_to_close = max(0.0, round(k * amt_step, amt_prec))
            else:
                amount_to_close = float(f"{amount_to_close:.{amt_prec}f}")
        except Exception:
            pass

        if amount_to_close <= 0:
            close_position(b.position_id, pnl_quote=0.0)
            try:
                send_telegram(f"ℹ️ {b.symbol}: amount below minimum step — closed logically.")
            except Exception:
                pass
            self.brackets.pop(b.symbol, None)
            return

        # enforce min notional
        try:
            m_info = exchange.markets.get(b.symbol, {}) if hasattr(exchange,'markets') else {}
            limits_info = m_info.get('limits', {}) or {}
            # Намалява се минималната номинална стойност за затваряне до 0,
            # така че дори много малки позиции да могат да бъдат продадени.
            min_notional_close = _safe_float((limits_info.get('cost') or {}).get('min'), 0.0)
        except Exception:
            # По подразбиране позволяваме затваряне на всякакъв номинал (0)
            min_notional_close = 0.0

        notional_close = amount_to_close * _safe_float(b.entry_price, 0.0)
        if notional_close < min_notional_close:
            log(f"[SKIPPED SELL] {b.symbol}: notional={notional_close:.2f} < min_notional={min_notional_close:.2f}")
            try:
                send_telegram(f"⚠️ {b.symbol}: quantity {amount_to_close:.6f} below min notional — will retry next tick.")
            except Exception:
                pass
            return

        # place sell order
        try:
            order = None
            if hasattr(exchange, 'create_order'):
                order = exchange.create_order(b.symbol, 'market', 'sell', amount_to_close)
            elif hasattr(exchange, 'create_market_sell_order'):
                order = exchange.create_market_sell_order(b.symbol, amount_to_close)
            else:
                raise AttributeError("Exchange does not support market sell orders")
            price = float(order.get('average') or order.get('price') or b.entry_price)
        except Exception as e:
            # If the primary sell attempt fails (e.g., due to insufficient balance),
            # attempt to sell a reduced quantity (99% of available amount) and then
            # close the remainder logically.  This prevents repeated failures
            # while ensuring most of the position is closed on‑chain.
            log(f"[SELL error] {b.symbol}: {e}")
            try:
                send_telegram(f"❌ SELL failed {b.symbol}: {e}")
            except Exception:
                pass
            # compute fallback: 99% of the bracket amount or available free balance
            try:
                # use the last known free balance if available; otherwise fetch again
                fallback_free = float((exchange.fetch_balance().get('free') or {}).get(b.symbol.split('/')[0], 0.0))
            except Exception:
                fallback_free = 0.0
            # fall back to 99% of the minimum of bracket size and free balance
            fallback_amt = min(b.amount, fallback_free) * 0.99
            # if fallback_amt is still greater than zero, attempt to sell
            sold = False
            if fallback_amt > 0.0:
                # reuse step and precision from earlier
                try:
                    m_fallback = exchange.markets.get(b.symbol, {}) if hasattr(exchange,'markets') else {}
                    limits_fb = m_fallback.get('limits', {}) or {}
                    prec_fb   = m_fallback.get('precision', {}) or {}
                    amt_step_fb = float((limits_fb.get('amount') or {}).get('step') or 0.0)
                    amt_prec_fb = int(prec_fb.get('amount') or 8)
                except Exception:
                    amt_step_fb = 0.0
                    amt_prec_fb = 8
                # adjust fallback amount to step/precision
                try:
                    if amt_step_fb and amt_step_fb > 0:
                        k_fb = int(fallback_amt / amt_step_fb)
                        fallback_amt = max(0.0, round(k_fb * amt_step_fb, amt_prec_fb))
                    else:
                        fallback_amt = float(f"{fallback_amt:.{amt_prec_fb}f}")
                except Exception:
                    pass
                # compute fallback notional
                fallback_entry_price = _safe_float(b.entry_price, 0.0)
                fallback_notional = fallback_amt * fallback_entry_price
                # ensure fallback meets min_notional before placing
                try:
                    m_info_fb = m_fallback
                    limits_info_fb = m_info_fb.get('limits', {}) or {}
                    min_notional_fb = _safe_float((limits_info_fb.get('cost') or {}).get('min'), 5.0)
                except Exception:
                    min_notional_fb = 5.0
                if fallback_notional >= min_notional_fb:
                    try:
                        fb_order = None
                        if hasattr(exchange, 'create_order'):
                            fb_order = exchange.create_order(b.symbol, 'market', 'sell', fallback_amt)
                        elif hasattr(exchange, 'create_market_sell_order'):
                            fb_order = exchange.create_market_sell_order(b.symbol, fallback_amt)
                        else:
                            raise AttributeError("Exchange does not support market sell orders")
                        fb_price = float(fb_order.get('average') or fb_order.get('price') or fallback_entry_price)
                        # record fallback order and close logically
                        pnl_fb = (fb_price - fallback_entry_price) * fallback_amt
                        insert_order(b.symbol, 'sell', fallback_amt, fb_price, fb_order)
                        close_position(b.position_id, pnl_quote=pnl_fb)
                        try:
                            from .capital_manager import CapitalManager
                        except Exception:
                            from capital_manager import CapitalManager
                        try:
                            cm = CapitalManager()
                            full_notional_fb = fallback_entry_price * float(b.amount)
                            cm.update_after_trade(b.symbol, quote_delta=-full_notional_fb, pnl_quote=pnl_fb)
                        except Exception:
                            pass
                        log_trade(b.symbol, {"side":"sell","amount":fallback_amt,"price":fb_price})
                        try:
                            send_telegram(f"✅ (Fallback) Closed {b.symbol} @ {fb_price:.4f} — {reason}")
                        except Exception:
                            pass
                        sold = True
                    except Exception as fb_e:
                        log(f"[SELL fallback error] {b.symbol}: {fb_e}")
                        try:
                            send_telegram(f"⚠️ Fallback SELL failed {b.symbol}: {fb_e}")
                        except Exception:
                            pass
            # regardless of fallback success, logically close the bracket to prevent repeated errors
            try:
                # if we didn't sell anything, update DB with zero pnl
                if not sold:
                    close_position(b.position_id, pnl_quote=0.0)
                self.brackets.pop(b.symbol, None)
            except Exception:
                pass
            return

        pnl = (price - _safe_float(b.entry_price,0.0)) * amount_to_close
        insert_order(b.symbol, 'sell', amount_to_close, price, order)
        close_position(b.position_id, pnl_quote=pnl)

        try:
            from .capital_manager import CapitalManager
        except Exception:
            from capital_manager import CapitalManager
        try:
            cm = CapitalManager()
            full_notional = _safe_float(b.entry_price,0.0) * float(b.amount)
            cm.update_after_trade(b.symbol, quote_delta=-full_notional, pnl_quote=pnl)
        except Exception:
            pass

        log_trade(b.symbol, {"side":"sell","amount":amount_to_close,"price":price})
        try:
            send_telegram(f"✅ Closed {b.symbol} @ {price:.4f} — {reason}")
        except Exception:
            pass

        self.brackets.pop(b.symbol, None)

    def on_price(self, exchange, symbol: str) -> Optional[str]:
        b = self.brackets.get(symbol)
        if not b:
            return None

        # check min-notional: if remaining free balance < min notional, close logically
        try:
            base, _ = symbol.split('/')
            bal = exchange.fetch_balance() or {}
            free_bal = float((bal.get('free') or {}).get(base, 0.0))
            m_info = exchange.markets.get(symbol, {}) if hasattr(exchange,'markets') else {}
            limits_info = m_info.get('limits', {}) or {}
            # Намалява се минималната номинална стойност за затваряне до 0
            min_notional_close = _safe_float((limits_info.get('cost') or {}).get('min'), 0.0)
            approx_price = _safe_float(b.entry_price, 0.0)
            if free_bal * approx_price < min_notional_close:
                self._close_long(exchange, b, reason="minNotional")
                return "minNotional"
        except Exception:
            pass

        # fetch current price
        try:
            t = exchange.fetch_ticker(symbol)
            safe_entry = _safe_float(b.entry_price, 0.0)
            price = _safe_price_from_ticker(t, fallback=safe_entry)
        except Exception as e:
            log(f"[on_price] {symbol} price error: {e}")
            return None
        if price <= 0:
            return None

        # update trailing stop
        if b.trail_pct:
            if price > b.highest:
                b.highest = price
            trail_stop = b.highest * (1 - b.trail_pct)
            if b.sl_price is None or trail_stop > b.sl_price:
                b.sl_price = trail_stop

        # partial take profit if tp1_pct set
        if b.tp1_pct and not b.tp1_done and price >= safe_entry * (1 + b.tp1_pct):
            part = b.amount * b.partial_ratio
            insert_fill(b.symbol, 'sell', part, price, order_id=None)
            b.amount -= part
            b.tp1_done = True
            log(f"[TP1] partial {b.symbol} amount={part:.6f} price={price:.4f}")
            try:
                send_telegram(f"🎯 TP1 {b.symbol}: sold {part:.6f} @ {price:.4f}")
            except Exception:
                pass

        # take profit
        if b.tp_price and price >= b.tp_price:
            self._close_long(exchange, b, reason="TP")
            return "TP"

        # stop loss
        if b.sl_price and price <= b.sl_price:
            self._close_long(exchange, b, reason="SL/Trail")
            return "SL/Trail"

        return None

    def on_tick(self, exchange, symbols: List[str]|None=None) -> None:
        to_check = list(self.brackets.keys()) if symbols is None else [s for s in symbols if s in self.brackets]
        for sym in to_check:
            try:
                self.on_price(exchange, sym)
            except Exception as e:
                log(f"[tick] error {sym}: {e}")

def _base_ccy(symbol: str) -> str:
    try:
        return symbol.split("/")[0]
    except Exception:
        return symbol

def snapshot_holdings(exchange) -> dict:
    try:
        bal   = exchange.fetch_balance() or {}
        free  = bal.get("free") or {}
        total = bal.get("total") or {}
        return {"free": dict(free), "total": dict(total)}
    except Exception as e:
        log(f"[reconcile] balances error: {e}")
        return {"free": {}, "total": {}}

def reconcile_open_positions(exchange, positions: dict, *, close_in_db: bool=True, eps: float=1e-12) -> None:
    """
    Syncs local position dict with actual Binance holdings. If base asset quantity is ~0,
    marks the position closed (and optionally closes in DB).
    """
    try:
        bal   = exchange.fetch_balance() or {}
        free  = bal.get("free") or {}
        total = bal.get("total") or {}

        try:
            from .storage import get_open_positions, close_position
            db_open = {r["symbol"]: r for r in get_open_positions()}
        except Exception:
            db_open = {}
            close_position = None  # type: ignore

        for sym in list(positions.keys()):
            base = _base_ccy(sym)
            amt_total = float(total.get(base, 0.0))
            amt_free  = float(free.get(base, 0.0))
            amt = max(amt_total, amt_free)
            if amt <= eps:
                row = db_open.get(sym)
                if close_in_db and row and close_position:
                    try:
                        close_position(int(row["id"]), pnl_quote=0.0)
                    except Exception as e:
                        log(f"[reconcile] close_position DB error {sym}: {e}")
                positions.pop(sym, None)
                log(f"[reconcile] closed locally after manual exit: {sym}")
    except Exception as e:
        log(f"[reconcile] error: {e}")

# ---------------------------------------------------------------------------
# Additional helpers and wrappers for higher-level bot integration.
#
# The following global `_TM` and accessor `_get_tm()` implement a lazily
# instantiated shared TradeManager instance.  This allows external modules
# (e.g. ``main_bot.py``) to share a single TradeManager without having to
# manage its lifecycle.  Wrapper functions for opening trades, monitoring
# positions and sending reports are also provided for convenience.

# global shared trade manager instance (initialised on first use)
_TM: Optional['TradeManager'] = None

def _get_tm() -> 'TradeManager':
    """Lazily instantiate and return a shared ``TradeManager``."""
    global _TM
    if _TM is None:
        _TM = TradeManager()
    return _TM

# Initialise the shared instance on module load for direct imports
_tm_instance: 'TradeManager' = _get_tm()

def execute_trade(
    exchange,
    symbol: str,
    probability: Optional[float] = None,
    threshold: float = 0.0,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    trail_pct: Optional[float] = None,
    *,
    partial_ratio: float = 0.5,
    tp1_pct: Optional[float] = None,
    df: Optional[Any] = None,
    **kwargs: Any,
) -> Optional[Bracket]:
    """
    Open a long position if ``probability`` >= ``threshold``.

    This helper uses the ``CapitalManager`` to size the quote allocation and then calls
    :meth:`TradeManager.open_long`. If the quote allocation is zero or an error occurs,
    returns ``None``.

    Parameters
    ----------
    exchange
        The exchange client used to place orders.
    symbol : str
        The trading pair to open a position on.
    probability : float
        The model probability associated with the trade signal.
    threshold : float
        The probability threshold that must be exceeded to trigger a trade.
    tp_pct : float
        Take‑profit percentage relative to the entry price.
    sl_pct : float
        Stop‑loss percentage relative to the entry price.
    trail_pct : Optional[float]
        Optional trailing stop percentage.
    partial_ratio : float, default 0.5
        Portion of the position to close on the first take‑profit (tp1) event.
    tp1_pct : Optional[float]
        Optional percentage profit at which to partially close the position (tp1).  If ``None``, no partial profit is taken.
    df : Optional[Any], default None
        Optional keyword argument for backwards compatibility.  Some callers may pass a pandas DataFrame or other
        ancillary data under the name ``df``; this parameter is unused by ``execute_trade`` and is accepted only to
        absorb extraneous arguments without raising a ``TypeError``.
    **kwargs : Any
        Additional keyword arguments are ignored.  They are accepted to maintain backwards compatibility with older
        caller signatures that may supply extra options.
    """
    try:
        # Ignore unused parameters
        _ = df  # Prevent unused variable warning
        # If any of the critical parameters are missing, do not place a trade.  Some callers may
        # inadvertently omit ``probability``, ``tp_pct`` or ``sl_pct``; treat these as a no‑op.
        if probability is None or tp_pct is None or sl_pct is None:
            return None
        if probability < threshold:
            return None
        try:
            from .capital_manager import CapitalManager  # type: ignore
        except Exception:
            from capital_manager import CapitalManager  # type: ignore
        cm = CapitalManager()
        quote = cm.size_quote(
            exchange=exchange,
            symbol=symbol,
            probability=probability,
            tp=tp_pct,
            sl=sl_pct,
        )
        if not quote or quote <= 0.0:
            return None
        tm = _get_tm()
        return tm.open_long(
            exchange=exchange,
            symbol=symbol,
            quote_capital=quote,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            trail_pct=trail_pct,
            partial_ratio=partial_ratio,
            tp1_pct=tp1_pct,
        )
    except Exception as e:
        log(f"[execute_trade] error {symbol}: {e}")
        try:
            send_telegram(f"🚨 Trade error {symbol}: {e}")
        except Exception:
            pass
        return None

def monitor_positions(exchange, symbols: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
    """Tick all active positions to evaluate TP/SL/trailing stops.

    Accepts optional extra positional/keyword arguments for backwards compatibility; any
    additional arguments beyond ``symbols`` are ignored.
    """
    try:
        tm = _get_tm()
        tm.on_tick(exchange, symbols)
    except Exception as e:
        log(f"[monitor] error: {e}")

def send_periodic_report(exchange, symbols: Optional[List[str]] = None, *args: Any, **kwargs: Any) -> None:
    """Send a summary of open positions via Telegram.

    Accepts optional extra positional/keyword arguments for backwards compatibility; any
    additional arguments beyond ``symbols`` are ignored.
    """
    try:
        tm = _get_tm()
        positions: Dict[str, Dict[str, float]] = {}
        for sym, b in tm.brackets.items():
            if symbols and sym not in symbols:
                continue
            try:
                positions[sym] = {
                    "entry": _safe_float(b.entry_price, 0.0),
                    "tp": float(b.tp_price) if b.tp_price else 0.0,
                    "sl": float(b.sl_price) if b.sl_price else 0.0,
                }
            except Exception:
                positions[sym] = {}
        # attempt to import formatter
        try:
            from .telegram_utils import format_status_summary  # type: ignore
        except Exception:
            try:
                from telegram_utils import format_status_summary  # type: ignore
            except Exception:
                def format_status_summary(pos: dict) -> str:
                    if not pos:
                        return "📊 Няма отворени позиции."
                    summary = "📊 Активни позиции:\n"
                    for s, d in pos.items():
                        try:
                            summary += f"— {s.split('/')[0]} @ {d['entry']:.4f} | TP: {d['tp']:.4f}, SL: {d['sl']:.4f}\n"
                        except Exception:
                            summary += f"— {s}: {d}\n"
                    return summary
        msg = format_status_summary(positions)
        send_telegram(msg)
    except Exception as e:
        log(f"[report] error: {e}")
