from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'
DB_PATH = DATA_DIR / 'dashboard.db'


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _hash_password(password: str, salt: str) -> str:
    payload = f'{salt}:{password}'.encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def init_database() -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            '''
        )
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset TEXT NOT NULL,
                max_rows INTEGER NOT NULL,
                status TEXT NOT NULL,
                f1 REAL NOT NULL,
                roc_auc REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                latency_ms_avg REAL NOT NULL,
                notes TEXT DEFAULT ''
            )
            '''
        )
        conn.commit()
    seed_default_user()


def seed_default_user() -> None:
    username = 'Durgesh'
    password = 'MiniProject@2026'
    with get_connection() as conn:
        existing = conn.execute('SELECT username FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            return
        legacy = conn.execute('SELECT username FROM users WHERE username = ?', ('admin',)).fetchone()
        if legacy:
            conn.execute('DELETE FROM users WHERE username = ?', ('admin',))
        salt = secrets.token_hex(8)
        conn.execute(
            'INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)',
            (
                username,
                _hash_password(password, salt),
                salt,
                'project-admin',
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def authenticate_user(username: str, password: str) -> tuple[bool, dict[str, Any] | None]:
    with get_connection() as conn:
        row = conn.execute('SELECT username, password_hash, salt, role, created_at FROM users WHERE username = ?', (username,)).fetchone()
    if row is None:
        return False, None
    expected = _hash_password(password, row['salt'])
    if expected != row['password_hash']:
        return False, None
    return True, dict(row)


def fetch_user(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute('SELECT username, role, created_at FROM users WHERE username = ?', (username,)).fetchone()
    return dict(row) if row is not None else None


def record_run(dataset: str, max_rows: int, status: str, metrics: dict[str, Any], notes: str = '') -> None:
    with get_connection() as conn:
        conn.execute(
            '''
            INSERT INTO run_history (
                timestamp, dataset, max_rows, status, f1, roc_auc, precision_score, recall_score, latency_ms_avg, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                datetime.now(timezone.utc).isoformat(),
                dataset,
                max_rows,
                status,
                float(metrics.get('f1', 0.0)),
                float(metrics.get('roc_auc', 0.0)),
                float(metrics.get('precision', 0.0)),
                float(metrics.get('recall', 0.0)),
                float(metrics.get('latency_ms_avg', 0.0)),
                notes,
            ),
        )
        conn.commit()


def fetch_run_history(limit: int | None = None) -> list[dict[str, Any]]:
    query = 'SELECT * FROM run_history ORDER BY timestamp DESC'
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += ' LIMIT ?'
        params = (limit,)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def fetch_users() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute('SELECT username, role, created_at FROM users ORDER BY created_at ASC').fetchall()
    return [dict(row) for row in rows]
