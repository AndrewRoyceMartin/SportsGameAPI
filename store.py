from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("PICKS_DB_PATH", "picks.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            match_date TEXT,
            match_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            league TEXT,
            country TEXT,
            market TEXT,
            selection TEXT NOT NULL,
            odds_decimal REAL,
            implied_prob REAL,
            model_prob REAL,
            edge REAL,
            ev_per_unit REAL,
            home_elo REAL,
            away_elo REAL,
            result TEXT,
            profit_loss REAL,
            is_experimental INTEGER DEFAULT 0,
            meta TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(match_date);
        CREATE INDEX IF NOT EXISTS idx_picks_selection ON picks(selection);
    """)
    _migrate(conn)
    conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(picks)")
    columns = {row[1] for row in cursor.fetchall()}
    if "is_experimental" not in columns:
        conn.execute("ALTER TABLE picks ADD COLUMN is_experimental INTEGER DEFAULT 0")
        conn.commit()

    idx_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_picks_unique'"
    ).fetchall()
    if not idx_rows:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_picks_unique "
            "ON picks(match_date, home_team, away_team, selection, odds_decimal)"
        )
        conn.commit()


def save_picks(picks: List[Dict[str, Any]]) -> int:
    init_db()
    conn = _connect()
    inserted = 0
    for p in picks:
        before = conn.total_changes
        conn.execute(
            """INSERT OR IGNORE INTO picks
               (match_date, match_time, home_team, away_team, league, country,
                market, selection, odds_decimal, implied_prob, model_prob,
                edge, ev_per_unit, home_elo, away_elo, is_experimental, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p.get("date", ""),
                p.get("time", ""),
                p.get("home_team", ""),
                p.get("away_team", ""),
                p.get("league", ""),
                p.get("country", ""),
                p.get("market", ""),
                p.get("selection", ""),
                p.get("odds_decimal"),
                p.get("implied_prob"),
                p.get("model_prob"),
                p.get("edge"),
                p.get("ev_per_unit"),
                p.get("home_elo"),
                p.get("away_elo"),
                p.get("is_experimental", 0),
                json.dumps({
                    k: v for k, v in p.items()
                    if k not in (
                        "date", "time", "home_team", "away_team", "league",
                        "country", "market", "selection", "odds_decimal",
                        "implied_prob", "model_prob", "edge", "ev_per_unit",
                        "home_elo", "away_elo", "is_experimental",
                    )
                }),
            ),
        )
        if conn.total_changes > before:
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


def get_recent_picks(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM picks ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_result(pick_id: int, result: str, profit_loss: float) -> None:
    init_db()
    conn = _connect()
    conn.execute(
        "UPDATE picks SET result = ?, profit_loss = ? WHERE id = ?",
        (result, profit_loss, pick_id),
    )
    conn.commit()
    conn.close()
