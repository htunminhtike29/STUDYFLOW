"""SQLite repository. A single connection and transaction per operation."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from migrations import migrate
from models import Task, hours

DEFAULT_PATH = Path(__file__).with_name("studyflow.db")
SAMPLES = (
    ("Data Structures", 1, 5, 5.0),
    ("Artificial Intelligence", 3, 4, 6.0),
    ("Statistics", 6, 3, 4.0),
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
    password_hash TEXT NOT NULL,
    daily_limit REAL NOT NULL DEFAULT 3.0 CHECK(daily_limit > 0 AND daily_limit <= 24),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject_name TEXT NOT NULL CHECK(length(trim(subject_name)) > 0),
    deadline TEXT NOT NULL CHECK(length(deadline)=10 AND date(deadline, '+0 days') IS deadline),
    difficulty INTEGER NOT NULL CHECK(difficulty IN (1,2,3,4,5)),
    total_hours REAL NOT NULL CHECK(total_hours > 0),
    completed_hours REAL NOT NULL DEFAULT 0.0 CHECK(completed_hours BETWEEN 0 AND total_hours),
    created_at TEXT NOT NULL,
    sample_key INTEGER CHECK(sample_key BETWEEN 1 AND 3),
    UNIQUE(student_id, sample_key)
);
CREATE INDEX IF NOT EXISTS idx_tasks_student_deadline ON tasks(student_id, deadline);
"""


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_task(name, deadline, difficulty, estimate):
    name = name.strip()
    if not 1 <= len(name) <= 160:
        raise ValueError("Enter a subject or task name of 1–160 characters.")
    if type(deadline) is not date:
        raise ValueError("Choose a valid calendar date.")
    if type(difficulty) is not int or difficulty not in range(1, 6):
        raise ValueError("Choose a difficulty from 1 to 5.")
    return name, deadline.isoformat(), difficulty, float(hours(estimate))


class Database:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path

    @contextmanager
    def connection(self, *, write=False):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                if write:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
        finally:
            conn.close()

    def initialize(self):
        with self.connection() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] == 1:
                return
            conn.executescript(SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            migrate(conn)

    @staticmethod
    def _seed(conn, sid, today):
        conn.executemany(
            """INSERT INTO tasks(student_id,subject_name,deadline,difficulty,total_hours,
               created_at,sample_key,estimated_hours,original_difficulty,creation_days) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            [(sid, name, (today + timedelta(days=offset)).isoformat(), diff, total,
              timestamp(), key, total, diff, offset) for key, (name, offset, diff, total) in enumerate(SAMPLES, 1)],
        )

    def create_account(self, sid, name, encoded_password, limit, today):
        with self.connection(write=True) as conn:
            if conn.execute("SELECT 1 FROM students WHERE student_id=?", (sid,)).fetchone():
                raise ValueError("This student ID is already registered.")
            conn.execute("INSERT INTO students VALUES(?,?,?,?,?)",
                         (sid, name, encoded_password, float(hours(limit, daily=True)), timestamp()))
            self._seed(conn, sid, today)

    def samples(self, sid, *, restore, today):
        with self.connection(write=True) as conn:
            conn.execute("DELETE FROM tasks WHERE student_id=? AND sample_key IS NOT NULL", (sid,))
            if restore:
                self._seed(conn, sid, today)

    def student(self, sid):
        with self.connection() as conn:
            return conn.execute("SELECT student_id,name,daily_limit FROM students WHERE student_id=?",
                                (sid,)).fetchone()

    def password_hash(self, sid):
        with self.connection() as conn:
            row = conn.execute("SELECT password_hash FROM students WHERE student_id=?", (sid,)).fetchone()
            return row[0] if row else None

    def set_limit(self, sid, value):
        with self.connection(write=True) as conn:
            conn.execute("UPDATE students SET daily_limit=? WHERE student_id=?",
                         (float(hours(value, daily=True)), sid))

    def tasks(self, sid):
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE student_id=? ORDER BY deadline,task_id", (sid,))
            return [Task(r["task_id"], r["subject_name"], date.fromisoformat(r["deadline"]), r["difficulty"],
                         Decimal(str(r["total_hours"])), Decimal(str(r["actual_hours"])),
                         Decimal(str(r["estimated_hours"])),
                         Decimal(str(r["predicted_hours"])) if r["predicted_hours"] is not None else None,
                         bool(r["finished"]), r["budget_source"], r["model_version"]) for r in rows]

    def add_task(self, sid, name, deadline, difficulty, estimate, *, budget=None, predicted=None,
                 version=None, source="Original estimate", today=None):
        fields = validate_task(name, deadline, difficulty, estimate)
        predicted = float(hours(predicted)) if predicted is not None else None
        if bool(version) != (predicted is not None):
            raise ValueError("A prediction requires its model version.")
        if source == "Original estimate":
            accepted = fields[-1]
        elif source == "ML recommendation" and predicted is not None:
            accepted = predicted
        elif source == "Custom budget":
            accepted = float(hours(budget))
        else:
            raise ValueError("Choose an available budget source.")
        with self.connection(write=True) as conn:
            return conn.execute("""INSERT INTO tasks(student_id,subject_name,deadline,difficulty,total_hours,
                                created_at,estimated_hours,original_difficulty,creation_days,predicted_hours,
                                model_version,budget_source,creation_verified) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                                (sid, *fields[:3], accepted, timestamp(), fields[-1], difficulty,
                                 (deadline-(today or date.today())).days, predicted, version, source)).lastrowid

    @staticmethod
    def _owned(conn, sid, task_id):
        row = conn.execute("SELECT * FROM tasks WHERE student_id=? AND task_id=?",
                           (sid, task_id)).fetchone()
        if row is None:
            raise ValueError("This task is no longer available. Refresh the planner.")
        return row

    def edit_task(self, sid, task_id, name, deadline, difficulty, estimate):
        fields = validate_task(name, deadline, difficulty, estimate)
        with self.connection(write=True) as conn:
            row = self._owned(conn, sid, task_id)
            if fields[-1] != row["total_hours"] and fields[-1] < row["actual_hours"]:
                raise ValueError("A revised budget must cover the time already logged.")
            conn.execute("""UPDATE tasks SET subject_name=?,deadline=?,difficulty=?,total_hours=?,
                         completed_hours=min(actual_hours,?),budget_source=? WHERE student_id=? AND task_id=?""",
                         (*fields, fields[-1], "Custom budget" if fields[-1] != row["total_hours"] else row["budget_source"], sid, task_id))

    def delete_task(self, sid, task_id):
        with self.connection(write=True) as conn:
            self._owned(conn, sid, task_id)
            conn.execute("DELETE FROM tasks WHERE student_id=? AND task_id=?", (sid, task_id))

    def log_progress(self, sid, task_id, duration):
        duration = hours(duration)
        with self.connection(write=True) as conn:
            row = self._owned(conn, sid, task_id)
            if row["finished"]:
                raise ValueError("This task is already complete.")
            actual = Decimal(str(row["actual_hours"])) + duration
            conn.execute("""INSERT INTO study_sessions(student_id,task_id,study_date,duration_hours,created_at)
                         VALUES(?,?,?,?,?)""", (sid, task_id, date.today().isoformat(), float(duration), timestamp()))
            conn.execute("""UPDATE tasks SET actual_hours=?,completed_hours=min(total_hours,?)
                         WHERE student_id=? AND task_id=?""", (float(actual), float(actual), sid, task_id))

    def revise_remaining(self, sid, task_id, additional):
        with self.connection(write=True) as conn:
            row = self._owned(conn, sid, task_id)
            if row["finished"]:
                raise ValueError("This task is already complete.")
            budget = hours(Decimal(str(row["actual_hours"])) + hours(additional))
            conn.execute("""UPDATE tasks SET total_hours=?,budget_source='Custom budget',completed_hours=actual_hours
                         WHERE student_id=? AND task_id=?""", (float(budget), sid, task_id))

    def finish(self, sid, task_id):
        with self.connection(write=True) as conn:
            row = self._owned(conn, sid, task_id)
            if row["finished"]:
                return
            if row["actual_hours"] <= 0:
                raise ValueError("Log the actual time spent before confirming completion.")
            when = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            conn.execute("UPDATE tasks SET finished=1,completed_at=? WHERE student_id=? AND task_id=?",
                         (when, sid, task_id))
            if row["sample_key"] is None:
                conn.execute("""INSERT INTO study_history(student_id,task_id,subject_name,difficulty,
                             days_until_deadline,estimated_hours,predicted_hours,actual_hours,completed_at,eligible)
                             VALUES(?,?,?,?,?,?,?,?,?,?)""",
                             (sid, task_id, row["subject_name"], row["original_difficulty"], row["creation_days"],
                              row["estimated_hours"], row["predicted_hours"], row["actual_hours"], when,
                              row["creation_verified"]))

    def history(self, sid):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("""SELECT * FROM study_history WHERE student_id=?
                   ORDER BY completed_at,history_id""", (sid,))]

    def sessions(self, sid):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("""SELECT t.subject_name,s.study_date,s.duration_hours
                   FROM study_sessions s JOIN tasks t USING(student_id,task_id) WHERE s.student_id=?
                   ORDER BY s.session_id DESC""", (sid,))]

    def save_model(self, sid, metadata, artifact, activate):
        with self.connection(write=True) as conn:
            if activate:
                conn.execute("UPDATE personal_models SET active=0 WHERE student_id=?", (sid,))
            conn.execute("INSERT INTO personal_models VALUES(?,?,?,?,?,?)",
                         (sid, metadata["version"], metadata["trained_at"], int(activate),
                          json.dumps(metadata, allow_nan=False), artifact))

    def active_model(self, sid):
        with self.connection() as conn:
            return conn.execute("SELECT metadata,artifact FROM personal_models WHERE student_id=? AND active=1",
                                (sid,)).fetchone()

    def model_reports(self, sid):
        with self.connection() as conn:
            return [(bool(r[0]), json.loads(r[1])) for r in conn.execute(
                "SELECT active,metadata FROM personal_models WHERE student_id=? ORDER BY rowid DESC", (sid,))]
