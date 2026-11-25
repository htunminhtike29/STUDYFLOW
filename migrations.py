"""Version 1 extends the original StudyFlow schema in one SQLite transaction."""
TASK_COLUMNS = (
    "estimated_hours REAL", "predicted_hours REAL CHECK(predicted_hours > 0)",
    "original_difficulty INTEGER", "creation_days INTEGER",
    "actual_hours REAL NOT NULL DEFAULT 0 CHECK(actual_hours >= 0)",
    "finished INTEGER NOT NULL DEFAULT 0 CHECK(finished IN (0,1))", "completed_at TEXT",
    "model_version TEXT", "budget_source TEXT NOT NULL DEFAULT 'Original estimate'",
    "creation_verified INTEGER NOT NULL DEFAULT 0 CHECK(creation_verified IN (0,1))",
)
TABLES = (
    """CREATE UNIQUE INDEX task_owner_key ON tasks(student_id,task_id)""",
    """CREATE TABLE study_sessions (
        session_id INTEGER PRIMARY KEY, student_id TEXT NOT NULL, task_id INTEGER NOT NULL,
        study_date TEXT NOT NULL, duration_hours REAL NOT NULL CHECK(duration_hours > 0),
        created_at TEXT NOT NULL, FOREIGN KEY(student_id,task_id)
        REFERENCES tasks(student_id,task_id) ON DELETE CASCADE)""",
    """CREATE TABLE study_history (
        history_id INTEGER PRIMARY KEY, student_id TEXT NOT NULL, task_id INTEGER NOT NULL UNIQUE,
        subject_name TEXT NOT NULL, difficulty INTEGER NOT NULL, days_until_deadline INTEGER NOT NULL,
        estimated_hours REAL NOT NULL, predicted_hours REAL, actual_hours REAL NOT NULL CHECK(actual_hours>0),
        completed_at TEXT NOT NULL, eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
        FOREIGN KEY(student_id,task_id) REFERENCES tasks(student_id,task_id) ON DELETE CASCADE)""",
    """CREATE TABLE personal_models (
        student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
        version TEXT NOT NULL, trained_at TEXT NOT NULL, active INTEGER NOT NULL CHECK(active IN (0,1)),
        metadata TEXT NOT NULL, artifact BLOB NOT NULL, PRIMARY KEY(student_id, version))""",
    "CREATE UNIQUE INDEX one_active_model ON personal_models(student_id) WHERE active=1",
)


def migrate(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 1:
        return
    if version != 0:
        raise ValueError("This database requires a newer version of StudyFlow.")
    for column in TASK_COLUMNS:
        conn.execute("ALTER TABLE tasks ADD COLUMN " + column)
    # Legacy aggregates have no session dates or trustworthy original feature snapshots.
    conn.execute("""UPDATE tasks SET estimated_hours=total_hours, actual_hours=completed_hours,
                 original_difficulty=difficulty, finished=(completed_hours>=total_hours),
                 creation_days=CAST(julianday(deadline)-julianday(date(created_at,'localtime')) AS INTEGER)""")
    for statement in TABLES:
        conn.execute(statement)
    conn.execute("PRAGMA user_version=1")
