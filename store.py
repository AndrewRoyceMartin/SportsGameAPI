from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime, timezone
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

        CREATE TABLE IF NOT EXISTS elo_overrides (
            league TEXT PRIMARY KEY,
            k REAL,
            home_adv REAL,
            scale REAL,
            recency_half_life REAL,
            updated_at TEXT
        );
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
        try:
            conn.execute(
                "DELETE FROM picks WHERE rowid NOT IN ("
                "  SELECT MIN(rowid) FROM picks "
                "  GROUP BY match_date, home_team, away_team, selection, odds_decimal"
                ")"
            )
            conn.execute(
                "CREATE UNIQUE INDEX idx_picks_unique "
                "ON picks(match_date, home_team, away_team, selection, odds_decimal)"
            )
            conn.commit()
        except Exception:
            conn.rollback()


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
                json.dumps(
                    {
                        k: v
                        for k, v in p.items()
                        if k
                        not in (
                            "date",
                            "time",
                            "home_team",
                            "away_team",
                            "league",
                            "country",
                            "market",
                            "selection",
                            "odds_decimal",
                            "implied_prob",
                            "model_prob",
                            "edge",
                            "ev_per_unit",
                            "home_elo",
                            "away_elo",
                            "is_experimental",
                        )
                    }
                ),
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


def get_rolling_stats() -> Dict[str, Any]:
    init_db()
    conn = _connect()

    settled = conn.execute(
        "SELECT * FROM picks WHERE result IN ('Won', 'Lost', 'Void') ORDER BY created_at DESC"
    ).fetchall()
    settled = [dict(r) for r in settled]

    total = len(settled)
    won = sum(1 for r in settled if r["result"] == "Won")
    lost = sum(1 for r in settled if r["result"] == "Lost")
    voided = sum(1 for r in settled if r["result"] == "Void")
    total_pl = sum(r.get("profit_loss", 0) or 0 for r in settled)
    total_staked = sum(1 for r in settled if r["result"] in ("Won", "Lost"))
    roi = (total_pl / total_staked * 100) if total_staked > 0 else 0

    quality_tiers = {"A": {"won": 0, "total": 0}, "B": {"won": 0, "total": 0}, "C": {"won": 0, "total": 0}}
    conf_buckets = {"50-59%": {"won": 0, "total": 0}, "60-69%": {"won": 0, "total": 0}, "70%+": {"won": 0, "total": 0}}

    for r in settled:
        if r["result"] == "Void":
            continue
        meta = {}
        if r.get("meta"):
            try:
                meta = json.loads(r["meta"])
            except Exception:
                pass
        q = meta.get("quality", 0)
        if q >= 80:
            t = "A"
        elif q >= 60:
            t = "B"
        else:
            t = "C"
        quality_tiers[t]["total"] += 1
        if r["result"] == "Won":
            quality_tiers[t]["won"] += 1

        mp = r.get("model_prob") or 0
        if mp >= 0.70:
            bucket = "70%+"
        elif mp >= 0.60:
            bucket = "60-69%"
        else:
            bucket = "50-59%"
        conf_buckets[bucket]["total"] += 1
        if r["result"] == "Won":
            conf_buckets[bucket]["won"] += 1

    conn.close()
    return {
        "total": total,
        "won": won,
        "lost": lost,
        "voided": voided,
        "total_pl": total_pl,
        "roi": roi,
        "quality_tiers": quality_tiers,
        "conf_buckets": conf_buckets,
    }


def save_elo_override(league: str, params: Dict[str, Any]) -> None:
    if not league:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO elo_overrides (league, k, home_adv, scale, recency_half_life, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(league) DO UPDATE SET
                k=excluded.k,
                home_adv=excluded.home_adv,
                scale=excluded.scale,
                recency_half_life=excluded.recency_half_life,
                updated_at=excluded.updated_at
            """,
            (league, params.get("k"), params.get("home_adv"),
             params.get("scale"), params.get("recency_half_life"), now),
        )
        conn.commit()
    finally:
        conn.close()


def load_elo_override(league: str) -> Optional[Dict[str, Any]]:
    if not league:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT k, home_adv, scale, recency_half_life, updated_at FROM elo_overrides WHERE league = ?",
            (league,),
        ).fetchone()
        if not row:
            return None
        k, home_adv, scale, recency_half_life, updated_at = row["k"], row["home_adv"], row["scale"], row["recency_half_life"], row["updated_at"]
        out: Dict[str, Any] = {}
        if k is not None:
            out["k"] = float(k)
        if home_adv is not None:
            out["home_adv"] = float(home_adv)
        if scale is not None:
            out["scale"] = float(scale)
        if recency_half_life is not None:
            out["recency_half_life"] = float(recency_half_life)
        out["_updated_at"] = updated_at
        return out
    finally:
        conn.close()


def load_all_elo_overrides() -> Dict[str, Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT league, k, home_adv, scale, recency_half_life, updated_at FROM elo_overrides"
        ).fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            params: Dict[str, Any] = {}
            if row["k"] is not None:
                params["k"] = float(row["k"])
            if row["home_adv"] is not None:
                params["home_adv"] = float(row["home_adv"])
            if row["scale"] is not None:
                params["scale"] = float(row["scale"])
            if row["recency_half_life"] is not None:
                params["recency_half_life"] = float(row["recency_half_life"])
            params["_updated_at"] = row["updated_at"]
            result[str(row["league"])] = params
        return result
    finally:
        conn.close()
