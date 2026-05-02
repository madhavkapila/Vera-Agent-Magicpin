"""
database.py — Vera Message Engine
Embedded SQLite state management matching the exact judge harness contract.

Schema:
  contexts  — (scope, context_id) composite PK, version-gated upserts
  conversations — append-only turn log keyed by conversation_id
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("vera.database")

DB_PATH = os.getenv("VERA_DB_PATH", "vera_state.db")

_conn: Optional[sqlite3.Connection] = None


def get_db() -> sqlite3.Connection:
    """Return the singleton database connection, initializing if needed."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _init_schema(_conn)
        _conn.execute("DELETE FROM contexts")
        _conn.execute("DELETE FROM conversations")
        _conn.commit()
        logger.info("Database initialized at %s", DB_PATH)
    return _conn


def close_db():
    """Gracefully close the database connection."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
        logger.info("Database connection closed")


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS contexts (
            scope       TEXT NOT NULL,
            context_id  TEXT NOT NULL,
            version     INTEGER NOT NULL DEFAULT 0,
            payload     TEXT NOT NULL DEFAULT '{}',
            delivered_at TEXT,
            stored_at   TEXT NOT NULL,
            PRIMARY KEY (scope, context_id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            turn_number     INTEGER NOT NULL DEFAULT 0,
            role            TEXT NOT NULL,
            message         TEXT NOT NULL DEFAULT '',
            timestamp       TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conv_id ON conversations(conversation_id);
    """)
    conn.commit()


# ─── Context CRUD ─────────────────────────────────────────────────────────────

def upsert_context(
    scope: str,
    context_id: str,
    version: int,
    payload: Dict[str, Any],
    delivered_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Idempotent context upsert matching judge contract:
    - If incoming version > stored version → replace atomically, return accepted=True
    - If incoming version <= stored version → return accepted=False + 409 (stale)
    - If context_id is new → insert, return accepted=True
    """
    db = get_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    row = db.execute(
        "SELECT version FROM contexts WHERE scope = ? AND context_id = ?",
        (scope, context_id),
    ).fetchone()

    if row is not None:
        current_version = row["version"]
        if version <= current_version:
            # Stale or duplicate — return 409 per judge contract
            return {
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
                "status_code": 409,
            }

    payload_json = json.dumps(payload, ensure_ascii=False)

    db.execute(
        """
        INSERT INTO contexts (scope, context_id, version, payload, delivered_at, stored_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope, context_id) DO UPDATE SET
            version = excluded.version,
            payload = excluded.payload,
            delivered_at = excluded.delivered_at,
            stored_at = excluded.stored_at
        """,
        (scope, context_id, version, payload_json, delivered_at, now),
    )
    db.commit()

    ack_id = f"ack_{context_id}_v{version}"
    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": now,
        "status_code": 200,
    }


def get_context(scope: str, context_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single context entry."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM contexts WHERE scope = ? AND context_id = ?",
        (scope, context_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "scope": row["scope"],
        "context_id": row["context_id"],
        "version": row["version"],
        "payload": json.loads(row["payload"]),
    }


def get_all_contexts_by_scope(scope: str) -> List[Dict[str, Any]]:
    """Get all contexts for a given scope."""
    db = get_db()
    rows = db.execute(
        "SELECT context_id, version, payload FROM contexts WHERE scope = ?",
        (scope,),
    ).fetchall()
    return [
        {
            "context_id": r["context_id"],
            "version": r["version"],
            "payload": json.loads(r["payload"]),
        }
        for r in rows
    ]


def count_contexts() -> Dict[str, int]:
    """Count contexts per scope — used by /v1/healthz."""
    db = get_db()
    rows = db.execute(
        "SELECT scope, COUNT(*) as cnt FROM contexts GROUP BY scope"
    ).fetchall()
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for r in rows:
        counts[r["scope"]] = r["cnt"]
    return counts


def wipe_all():
    """Teardown — wipe all state."""
    db = get_db()
    db.execute("DELETE FROM contexts")
    db.execute("DELETE FROM conversations")
    db.commit()
    logger.info("All state wiped (teardown)")


# ─── Conversation CRUD ────────────────────────────────────────────────────────

def append_turn(
    conversation_id: str,
    turn_number: int,
    role: str,
    message: str,
):
    """Append a turn to a conversation."""
    db = get_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    db.execute(
        """
        INSERT INTO conversations (conversation_id, turn_number, role, message, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conversation_id, turn_number, role, message, now),
    )
    db.commit()


def get_conversation(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve all turns for a conversation in chronological order."""
    db = get_db()
    rows = db.execute(
        """
        SELECT turn_number, role, message, timestamp
        FROM conversations
        WHERE conversation_id = ?
        ORDER BY turn_number ASC, id ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [
        {
            "turn_number": r["turn_number"],
            "role": r["role"],
            "message": r["message"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]


def conversation_exists(conversation_id: str) -> bool:
    """Check if a conversation already has turns logged."""
    db = get_db()
    row = db.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ? LIMIT 1",
        (conversation_id,),
    ).fetchone()
    return row is not None
