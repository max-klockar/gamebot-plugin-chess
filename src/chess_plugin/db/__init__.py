from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    skill_band TEXT NOT NULL DEFAULT 'intermediate',
    skill_level INTEGER NOT NULL DEFAULT 5,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER,
    skill_level INTEGER NOT NULL,
    result TEXT,
    pgn TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    FOREIGN KEY (profile_id) REFERENCES profiles(id)
);

CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    ply INTEGER NOT NULL,
    uci TEXT NOT NULL,
    san TEXT NOT NULL,
    by_engine INTEGER NOT NULL DEFAULT 0,
    played_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(id)
);
"""


@dataclass
class Profile:
    id: int
    name: str
    skill_band: str
    skill_level: int


@dataclass
class GameRecord:
    id: int
    profile_id: int | None
    skill_level: int
    result: str | None
    pgn: str | None


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._ensure_guest()

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_guest(self) -> None:
        row = self._conn.execute(
            "SELECT id FROM profiles WHERE name = ?", ("Guest",)
        ).fetchone()
        if row is None:
            self.create_profile("Guest", "intermediate", 5)

    def create_profile(self, name: str, skill_band: str, skill_level: int) -> Profile:
        cur = self._conn.execute(
            "INSERT INTO profiles (name, skill_band, skill_level, created_at) VALUES (?, ?, ?, ?)",
            (name, skill_band, skill_level, self._now()),
        )
        self._conn.commit()
        return Profile(cur.lastrowid, name, skill_band, skill_level)

    def list_profiles(self) -> list[Profile]:
        rows = self._conn.execute(
            "SELECT id, name, skill_band, skill_level FROM profiles ORDER BY name"
        ).fetchall()
        return [Profile(r["id"], r["name"], r["skill_band"], r["skill_level"]) for r in rows]

    def get_profile(self, profile_id: int) -> Profile | None:
        r = self._conn.execute(
            "SELECT id, name, skill_band, skill_level FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
        if r is None:
            return None
        return Profile(r["id"], r["name"], r["skill_band"], r["skill_level"])

    def update_profile_skill(self, profile_id: int, skill_band: str, skill_level: int) -> None:
        self._conn.execute(
            "UPDATE profiles SET skill_band = ?, skill_level = ? WHERE id = ?",
            (skill_band, skill_level, profile_id),
        )
        self._conn.commit()

    def start_game(self, profile_id: int | None, skill_level: int) -> GameRecord:
        cur = self._conn.execute(
            "INSERT INTO games (profile_id, skill_level, started_at) VALUES (?, ?, ?)",
            (profile_id, skill_level, self._now()),
        )
        self._conn.commit()
        return GameRecord(cur.lastrowid, profile_id, skill_level, None, None)

    def add_move(self, game_id: int, ply: int, uci: str, san: str, by_engine: bool) -> None:
        self._conn.execute(
            "INSERT INTO moves (game_id, ply, uci, san, by_engine, played_at) VALUES (?, ?, ?, ?, ?, ?)",
            (game_id, ply, uci, san, int(by_engine), self._now()),
        )
        self._conn.commit()

    def finish_game(self, game_id: int, result: str, pgn: str) -> None:
        self._conn.execute(
            "UPDATE games SET result = ?, pgn = ?, ended_at = ? WHERE id = ?",
            (result, pgn, self._now(), game_id),
        )
        self._conn.commit()

    def moves_for_game(self, game_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT ply, uci, san, by_engine FROM moves WHERE game_id = ? ORDER BY ply",
            (game_id,),
        ).fetchall()
