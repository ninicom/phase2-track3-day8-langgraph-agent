"""Checkpointer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = (database_url or "outputs/checkpoints.sqlite").replace("sqlite:///", "")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the saver can be reused across the web server's threads.
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
