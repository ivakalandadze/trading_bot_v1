"""
database.py — SQLite persistence layer.

Tables:
  signals       — every technique signal generated
  trades        — every trade opened / closed
  positions     — current open positions
  paper_account — paper trading account state
  scan_log      — scan run history
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional
import config


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent access
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.executescript("""
    -- Individual technique signals
    CREATE TABLE IF NOT EXISTS signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT    NOT NULL,
        symbol        TEXT    NOT NULL,
        asset_type    TEXT    NOT NULL,   -- 'stock' | 'crypto'
        technique     TEXT    NOT NULL,
        signal        TEXT    NOT NULL,   -- 'BUY' | 'SELL' | 'NEUTRAL'
        score         REAL    NOT NULL,
        confidence    REAL    NOT NULL,
        reasoning     TEXT    NOT NULL    -- JSON string
    );

    -- Trade records
    CREATE TABLE IF NOT EXISTS trades (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT    NOT NULL,
        symbol        TEXT    NOT NULL,
        asset_type    TEXT    NOT NULL,
        side          TEXT    NOT NULL,   -- 'BUY' | 'SELL'
        quantity      REAL    NOT NULL,
        entry_price   REAL    NOT NULL,
        exit_price    REAL,
        stop_loss     REAL,
        take_profit   REAL,
        pnl           REAL,
        status        TEXT    NOT NULL DEFAULT 'open',   -- 'open' | 'closed'
        mode          TEXT    NOT NULL,   -- 'paper' | 'live'
        llm_reasoning TEXT,
        techniques_summary TEXT          -- JSON snapshot of signals
    );

    -- Current open positions (denormalised for speed)
    CREATE TABLE IF NOT EXISTS positions (
        symbol        TEXT    PRIMARY KEY,
        asset_type    TEXT    NOT NULL,
        quantity      REAL    NOT NULL,
        entry_price   REAL    NOT NULL,
        current_price REAL    NOT NULL,
        stop_loss     REAL,
        take_profit   REAL,
        opened_at     TEXT    NOT NULL,
        trade_id      INTEGER,
        mode          TEXT    NOT NULL
    );

    -- Paper trading account
    CREATE TABLE IF NOT EXISTS paper_account (
        id            INTEGER PRIMARY KEY CHECK (id = 1),
        cash          REAL    NOT NULL,
        equity        REAL    NOT NULL,
        total_value   REAL    NOT NULL,
        updated_at    TEXT    NOT NULL
    );

    -- Scan history
    CREATE TABLE IF NOT EXISTS scan_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at    TEXT    NOT NULL,
        finished_at   TEXT,
        symbols_scanned INTEGER,
        signals_generated INTEGER,
        trades_executed INTEGER,
        errors        TEXT
    );
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────

def save_signal(symbol: str, asset_type: str, technique: str,
                signal: str, score: float, confidence: float,
                reasoning: dict) -> None:
    conn = _get_connection()
    conn.execute(
        """INSERT INTO signals
           (timestamp, symbol, asset_type, technique, signal, score, confidence, reasoning)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.utcnow().isoformat(), symbol, asset_type,
         technique, signal, round(score, 4), round(confidence, 4),
         json.dumps(reasoning))
    )
    conn.commit()
    conn.close()


def get_recent_signals(symbol: str, hours: int = 4) -> list[dict]:
    conn = _get_connection()
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT * FROM signals WHERE symbol=? AND timestamp>=? ORDER BY timestamp DESC",
        (symbol, cutoff)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Trades
# ─────────────────────────────────────────────────────────────────────────────

def open_trade(symbol: str, asset_type: str, side: str, quantity: float,
               entry_price: float, stop_loss: float, take_profit: float,
               mode: str, llm_reasoning: str = "",
               techniques_summary: dict = None) -> int:
    conn = _get_connection()
    cur = conn.execute(
        """INSERT INTO trades
           (timestamp, symbol, asset_type, side, quantity, entry_price,
            stop_loss, take_profit, status, mode, llm_reasoning, techniques_summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
        (datetime.utcnow().isoformat(), symbol, asset_type, side,
         quantity, entry_price, stop_loss, take_profit,
         mode, llm_reasoning, json.dumps(techniques_summary or {}))
    )
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def close_trade(trade_id: int, exit_price: float, pnl: float) -> None:
    conn = _get_connection()
    conn.execute(
        """UPDATE trades
           SET exit_price=?, pnl=?, status='closed'
           WHERE id=?""",
        (exit_price, round(pnl, 4), trade_id)
    )
    conn.commit()
    conn.close()


def get_open_trades(mode: str) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status='open' AND mode=?", (mode,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_history(mode: str, limit: int = 50) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM trades WHERE mode=? ORDER BY timestamp DESC LIMIT ?",
        (mode, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Positions
# ─────────────────────────────────────────────────────────────────────────────

def upsert_position(symbol: str, asset_type: str, quantity: float,
                    entry_price: float, current_price: float,
                    stop_loss: float, take_profit: float,
                    trade_id: int, mode: str) -> None:
    conn = _get_connection()
    conn.execute(
        """INSERT INTO positions
           (symbol, asset_type, quantity, entry_price, current_price,
            stop_loss, take_profit, opened_at, trade_id, mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET
             current_price=excluded.current_price,
             quantity=excluded.quantity""",
        (symbol, asset_type, quantity, entry_price, current_price,
         stop_loss, take_profit, datetime.utcnow().isoformat(), trade_id, mode)
    )
    conn.commit()
    conn.close()


def remove_position(symbol: str) -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
    conn.commit()
    conn.close()


def get_positions(mode: str) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM positions WHERE mode=?", (mode,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_position_price(symbol: str, current_price: float) -> None:
    conn = _get_connection()
    conn.execute(
        "UPDATE positions SET current_price=? WHERE symbol=?",
        (round(current_price, 6), symbol)
    )
    conn.commit()
    conn.close()


def update_position_stop(symbol: str, new_stop: float) -> None:
    conn = _get_connection()
    conn.execute(
        "UPDATE positions SET stop_loss=? WHERE symbol=?",
        (round(new_stop, 6), symbol)
    )
    conn.commit()
    conn.close()


def update_position_after_partial(symbol: str, new_quantity: float,
                                   new_stop: float, new_tp: float) -> None:
    """Update position in-place after a partial close."""
    conn = _get_connection()
    conn.execute(
        "UPDATE positions SET quantity=?, stop_loss=?, take_profit=? WHERE symbol=?",
        (round(new_quantity, 6), round(new_stop, 6), round(new_tp, 6), symbol)
    )
    conn.commit()
    conn.close()


def get_latest_signal_quality(symbol: str, hours: int = 3) -> tuple[float, float]:
    """
    Return (avg_score, avg_confidence) of BUY signals for symbol
    from the last `hours` hours. Used to determine partial-close fraction.
    """
    conn = _get_connection()
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    row = conn.execute(
        """SELECT AVG(score), AVG(confidence) FROM signals
           WHERE symbol=? AND timestamp>=? AND signal='BUY'""",
        (symbol, cutoff)
    ).fetchone()
    conn.close()
    if row and row[0] is not None:
        return float(row[0]), float(row[1])
    return 50.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Paper Account
# ─────────────────────────────────────────────────────────────────────────────

def init_paper_account(starting_cash: float) -> None:
    conn = _get_connection()
    conn.execute(
        """INSERT OR IGNORE INTO paper_account (id, cash, equity, total_value, updated_at)
           VALUES (1, ?, 0, ?, ?)""",
        (starting_cash, starting_cash, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_paper_account() -> dict:
    conn = _get_connection()
    row = conn.execute("SELECT * FROM paper_account WHERE id=1").fetchone()
    conn.close()
    return dict(row) if row else {}


def update_paper_account(cash: float, equity: float) -> None:
    conn = _get_connection()
    conn.execute(
        """UPDATE paper_account
           SET cash=?, equity=?, total_value=?, updated_at=?
           WHERE id=1""",
        (round(cash, 4), round(equity, 4),
         round(cash + equity, 4), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Scan Log
# ─────────────────────────────────────────────────────────────────────────────

def start_scan_log() -> int:
    conn = _get_connection()
    cur = conn.execute(
        "INSERT INTO scan_log (started_at) VALUES (?)",
        (datetime.utcnow().isoformat(),)
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def finish_scan_log(scan_id: int, symbols_scanned: int,
                    signals_generated: int, trades_executed: int,
                    errors: str = "") -> None:
    conn = _get_connection()
    conn.execute(
        """UPDATE scan_log
           SET finished_at=?, symbols_scanned=?, signals_generated=?,
               trades_executed=?, errors=?
           WHERE id=?""",
        (datetime.utcnow().isoformat(), symbols_scanned,
         signals_generated, trades_executed, errors, scan_id)
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────────────────────────────────────

def reset_portfolio(starting_cash: float) -> None:
    """Wipe all trading state and reset paper account to starting_cash."""
    conn = _get_connection()
    conn.execute("DELETE FROM positions")
    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM signals")
    conn.execute("DELETE FROM scan_log")
    conn.execute(
        """UPDATE paper_account
           SET cash=?, equity=0, total_value=?, updated_at=?
           WHERE id=1""",
        (starting_cash, starting_cash, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_daily_pnl(mode: str, date_iso: str) -> float:
    """Return sum of realised P&L for closed trades on the given date (YYYY-MM-DD)."""
    conn = _get_connection()
    row = conn.execute(
        """SELECT COALESCE(SUM(pnl), 0) FROM trades
           WHERE mode=? AND status='closed' AND DATE(timestamp)=?""",
        (mode, date_iso)
    ).fetchone()
    conn.close()
    return float(row[0])


def get_pnl_summary(mode: str) -> dict:
    conn = _get_connection()
    row = conn.execute(
        """SELECT
             COUNT(*)           AS total_trades,
             SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS winners,
             SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losers,
             ROUND(SUM(pnl), 2) AS total_pnl,
             ROUND(AVG(pnl), 2) AS avg_pnl,
             ROUND(MAX(pnl), 2) AS best_trade,
             ROUND(MIN(pnl), 2) AS worst_trade
           FROM trades
           WHERE status='closed' AND mode=?""",
        (mode,)
    ).fetchone()
    conn.close()
    return dict(row) if row else {}
