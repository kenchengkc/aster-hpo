"""Single-coordinator SQLite journal with transactional observation acceptance."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


class SQLiteStore:
    """Local filesystem store; a process lock excludes concurrent coordinators.

    Observations and attempts are normalized rather than rewriting all previous
    observations on every completion. An in-memory database implements the same
    state machine when persistence is not requested.
    """

    def __init__(self, path, manifest):
        self.lock = None
        self.db = None
        try:
            if path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.lock = open(str(path) + ".lock", "a+b")
                try:
                    if os.name == "nt":
                        import msvcrt

                        self.lock.seek(0)
                        self.lock.write(b"\0")
                        self.lock.flush()
                        self.lock.seek(0)
                        msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError("Experiment database is already in use") from exc
            self.db = sqlite3.connect(":memory:" if path is None else path, timeout=5)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (
                    seq INTEGER PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, trial_id INTEGER NOT NULL,
                    replication_index INTEGER NOT NULL, replication TEXT NOT NULL,
                    status TEXT NOT NULL, submitted_at REAL NOT NULL, finished_at REAL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS observations (
                    trial_id INTEGER NOT NULL, replication_index INTEGER NOT NULL,
                    attempt_id INTEGER NOT NULL UNIQUE, payload TEXT NOT NULL,
                    PRIMARY KEY (trial_id, replication_index)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, at REAL NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)
            previous = self.db.execute("SELECT payload FROM manifest WHERE id=1").fetchone()
            if previous and previous["payload"] != _json(manifest):
                raise ValueError("Experiment manifest differs; use a new database")
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO manifest VALUES (1, ?)", (_json(manifest),))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def _event(self, kind, payload):
        self.db.execute(
            "INSERT INTO events (at,kind,payload) VALUES (?,?,?)",
            (time.time(), kind, _json(payload)),
        )

    def _save(self, snapshot):
        state = {
            **snapshot,
            "trials": [
                {key: value for key, value in trial.items() if key != "observations"}
                for trial in snapshot["trials"]
            ],
        }
        decisions = state.pop("decisions")
        self.db.execute(
            "INSERT INTO state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
            (_json(state),),
        )
        previous = self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        self.db.executemany(
            "INSERT INTO decisions VALUES (?,?)",
            [(i, _json(d)) for i, d in enumerate(decisions) if i >= previous],
        )

    def save(self, snapshot):
        with self.db:
            self._save(snapshot)

    def load(self):
        saved = self.db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        if saved is None:
            return None
        state = json.loads(saved["payload"])
        observations = {}
        for row in self.db.execute(
            "SELECT * FROM observations ORDER BY trial_id,replication_index"
        ):
            observations.setdefault(row["trial_id"], []).append(json.loads(row["payload"]))
        for trial in state["trials"]:
            trial["observations"] = observations.get(trial["id"], [])
        state["decisions"] = [
            json.loads(row[0])
            for row in self.db.execute("SELECT payload FROM decisions ORDER BY seq")
        ]
        return state

    def recover(self, snapshot):
        """Account for work with no committed result after an unclean shutdown."""
        with self.db:
            rows = self.db.execute("SELECT id FROM attempts WHERE status='running'").fetchall()
            for row in rows:
                self.db.execute(
                    "UPDATE attempts SET status='interrupted',finished_at=?,error=? WHERE id=?",
                    (time.time(), "coordinator interrupted", row["id"]),
                )
                self._event("interrupted", {"attempt_id": row["id"]})
            self._save(snapshot)

    def attempt_count(self, trial_id, replication_index):
        return self.db.execute(
            "SELECT COUNT(*) FROM attempts WHERE trial_id=? AND replication_index=?",
            (trial_id, replication_index),
        ).fetchone()[0]

    def begin_attempt(self, trial_id, replication, snapshot):
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO attempts (trial_id,replication_index,replication,status,submitted_at) "
                "VALUES (?,?,?,'running',?)",
                (trial_id, replication["index"], _json(replication), time.time()),
            )
            attempt_id = cursor.lastrowid
            self._save(snapshot)
            self._event(
                "submit",
                {
                    "attempt_id": attempt_id,
                    "trial_id": trial_id,
                    "replication_index": replication["index"],
                },
            )
        return attempt_id

    def accept(self, attempt_id, observation, snapshot):
        with self.db:
            attempt = self.db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if attempt is None or attempt["status"] != "running":
                return False
            if _json(observation["replication"]) != attempt["replication"]:
                raise ValueError("Observation identity does not match its attempt")
            duplicate = self.db.execute(
                "SELECT 1 FROM observations WHERE trial_id=? AND replication_index=?",
                (attempt["trial_id"], attempt["replication_index"]),
            ).fetchone()
            if duplicate:
                return False
            self.db.execute(
                "INSERT INTO observations VALUES (?,?,?,?)",
                (attempt["trial_id"], attempt["replication_index"], attempt_id, _json(observation)),
            )
            self.db.execute(
                "UPDATE attempts SET status='accepted',finished_at=? WHERE id=?",
                (time.time(), attempt_id),
            )
            self._save(snapshot)
            self._event(
                "accept",
                {
                    "attempt_id": attempt_id,
                    "trial_id": attempt["trial_id"],
                    "replication_index": attempt["replication_index"],
                },
            )
        return True

    def fail_attempt(self, attempt_id, status, error, snapshot):
        if status not in {"interrupted", "infrastructure_failed"}:
            raise ValueError("Invalid failed-attempt status")
        with self.db:
            cursor = self.db.execute(
                "UPDATE attempts SET status=?,finished_at=?,error=? "
                "WHERE id=? AND status='running'",
                (status, time.time(), error, attempt_id),
            )
            if cursor.rowcount:
                self._event(status, {"attempt_id": attempt_id, "error": error})
            self._save(snapshot)

    def journal(self):
        attempts = []
        for row in self.db.execute("SELECT * FROM attempts ORDER BY id"):
            item = dict(row)
            item["replication"] = json.loads(item["replication"])
            attempts.append(item)
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in self.db.execute("SELECT * FROM events ORDER BY seq")
        ]
        return attempts, events
