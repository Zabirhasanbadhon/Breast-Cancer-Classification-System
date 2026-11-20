"""SQLite storage for classification history.

No personally identifiable information is stored - every record gets a generated
record ID such as ``BC-20261114-3F9A1C``.
"""
from __future__ import annotations

import csv
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import DATABASE_PATH, FEATURE_NAMES

BASE_COLUMNS = ["record_id", "created_at", "result", "probability", "prob_malignant", "model_name"]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Database:
    def __init__(self, path=DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _session(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        feature_cols = ",\n                ".join(f"{name} REAL NOT NULL" for name in FEATURE_NAMES)
        with self._session() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    result TEXT NOT NULL CHECK (result IN ('Benign', 'Malignant')),
                    class_code INTEGER NOT NULL,
                    probability REAL,
                    prob_malignant REAL,
                    model_name TEXT,
                    {feature_cols}
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)")

    @staticmethod
    def new_record_id() -> str:
        return f"BC-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"

    # ------------------------------------------------------------------ write
    def add_prediction(self, record: dict) -> str:
        """Insert one classification. ``record`` comes from ``PredictionResult.to_record()``."""
        values = record["values"]
        missing = [f for f in FEATURE_NAMES if f not in values]
        if missing:
            raise ValueError(f"Missing feature values: {missing}")
        columns = ["record_id", "created_at", "result", "class_code", "probability", "prob_malignant", "model_name"]
        columns += FEATURE_NAMES
        placeholders = ", ".join("?" for _ in columns)
        for _ in range(5):  # retry on the (extremely unlikely) record-id collision
            record_id = self.new_record_id()
            params = [
                record_id,
                record["created_at"],
                record["result"],
                int(record["class_code"]),
                record.get("probability"),
                record.get("prob_malignant"),
                record.get("model_name"),
            ] + [float(values[f]) for f in FEATURE_NAMES]
            try:
                with self._session() as conn:
                    conn.execute(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})", params)
                return record_id
            except sqlite3.IntegrityError as exc:
                if "record_id" not in str(exc):
                    raise
        raise RuntimeError("Could not generate a unique record ID")

    def delete_records(self, ids: list[int]) -> int:
        """Delete rows by their integer primary key. Returns the number removed."""
        ids = [int(i) for i in ids]
        removed = 0
        with self._session() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                cur = conn.execute(
                    f"DELETE FROM predictions WHERE id IN ({', '.join('?' for _ in chunk)})", chunk
                )
                removed += cur.rowcount
        return removed

    # ------------------------------------------------------------------ read
    def get_records(self, search: str = "", result_filter: str = "All", limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM predictions"
        where, params = [], []
        if result_filter in ("Benign", "Malignant"):
            where.append("result = ?")
            params.append(result_filter)
        term = (search or "").strip()
        if term:
            like = f"%{_escape_like(term)}%"
            where.append(
                "(record_id LIKE ? ESCAPE '\\' OR result LIKE ? ESCAPE '\\' "
                "OR created_at LIKE ? ESCAPE '\\' OR model_name LIKE ? ESCAPE '\\')"
            )
            params += [like] * 4
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._session() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def get_record(self, db_id: int) -> dict | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (int(db_id),)).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        with self._session() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def stats(self) -> dict:
        with self._session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(result = 'Benign'), 0) AS benign, "
                "COALESCE(SUM(result = 'Malignant'), 0) AS malignant FROM predictions"
            ).fetchone()
            latest = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT 1"
            ).fetchone()
        return {
            "total": int(row["total"]),
            "benign": int(row["benign"]),
            "malignant": int(row["malignant"]),
            "latest": dict(latest) if latest else None,
        }

    def timeline(self) -> list[tuple[str, int, int]]:
        """[(YYYY-MM-DD, benign_count, malignant_count)] ordered by day."""
        with self._session() as conn:
            rows = conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, "
                "SUM(result = 'Benign') AS benign, SUM(result = 'Malignant') AS malignant "
                "FROM predictions GROUP BY day ORDER BY day"
            ).fetchall()
        return [(r["day"], int(r["benign"]), int(r["malignant"])) for r in rows]

    # ------------------------------------------------------------------ export
    def export_csv(self, path, records: list[dict] | None = None) -> int:
        """Write history to CSV (all records unless ``records`` is given). Returns row count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if records is None:
            records = self.get_records()
        columns = BASE_COLUMNS + FEATURE_NAMES
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        return len(records)
