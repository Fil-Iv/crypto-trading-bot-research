# utils/storage.py
from __future__ import annotations
import os
import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timezone

DB_PATH = Path(os.getenv("BOT_DB_PATH", "data/bot.sqlite"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def init_db() -> None:
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals(
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prob REAL,
            threshold REAL,
            params_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders(
            ts TEXT NOT NULL,
            order_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT,
            amount REAL,
            price REAL,
            info_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fills(
            ts TEXT NOT NULL,
            order_id TEXT,
            trade_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT,
            amount REAL,
            price REAL,
            fee REAL,
            info_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics(
            ts TEXT NOT NULL,
            key TEXT NOT NULL,
            value REAL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty_open REAL NOT NULL,
            entry_price REAL NOT NULL,
            tp REAL,
            sl REAL,
            trailing REAL,
            status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN/CLOSED
            opened_ts TEXT NOT NULL,
            closed_ts TEXT,
            pnl_quote REAL DEFAULT 0
        );
        """)
        con.commit()

# ----------------- Insert helpers -----------------

def insert_signal(symbol: str, prob: float, threshold: float, params: Dict[str, Any]) -> None:
    with _conn() as con:
        con.execute("INSERT INTO signals VALUES (?,?,?,?,?)",
                    (_utcnow(), symbol, float(prob), float(threshold), json.dumps(params)))

def insert_order(symbol: str, side: str, amount: float, price: float, info: Dict[str, Any], order_id: Optional[str]=None) -> None:
    with _conn() as con:
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                    (_utcnow(), order_id, symbol, side, float(amount), float(price), json.dumps(info)))

def insert_fill(symbol: str, side: str, amount: float, price: float, fee: float=0.0, info: Optional[Dict[str, Any]]=None,
                order_id: Optional[str]=None, trade_id: Optional[str]=None) -> None:
    with _conn() as con:
        con.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?)",
                    (_utcnow(), order_id, trade_id, symbol, side, float(amount), float(price), float(fee), json.dumps(info or {})))

def insert_metric(key: str, value: float) -> None:
    with _conn() as con:
        con.execute("INSERT INTO metrics VALUES (?,?,?)", (_utcnow(), key, float(value)))

# ----------------- Positions -----------------

def upsert_open_position(symbol: str, side: str, qty_open: float, entry_price: float,
                         tp: Optional[float], sl: Optional[float], trailing: Optional[float]) -> int:
    """
    Creates a new open position row and returns its id.
    """
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO positions(symbol, side, qty_open, entry_price, tp, sl, trailing, status, opened_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (symbol, side, float(qty_open), float(entry_price),
              None if tp is None else float(tp),
              None if sl is None else float(sl),
              None if trailing is None else float(trailing),
              _utcnow()))
        return int(cur.lastrowid)

def close_position(position_id: int, pnl_quote: float=0.0) -> None:
    with _conn() as con:
        con.execute("""
            UPDATE positions SET status='CLOSED', closed_ts=?, pnl_quote=? WHERE id=?
        """, (_utcnow(), float(pnl_quote), int(position_id)))

def get_open_positions() -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute("SELECT id, symbol, side, qty_open, entry_price, tp, sl, trailing, opened_ts FROM positions WHERE status='OPEN'")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
