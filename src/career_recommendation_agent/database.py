from __future__ import annotations

import json
import os
import time
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from career_recommendation_agent.security import hash_password, verify_password


def database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def using_postgres() -> bool:
    return database_url().startswith(("postgres://", "postgresql://"))


def database_path() -> Path:
    configured = os.getenv("DATABASE_PATH", "data/career_agent.db")
    return Path(configured)


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def connect_postgres():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("PostgreSQL requires installing the psycopg dependency.") from exc
    return psycopg.connect(database_url(), row_factory=dict_row)


def init_database() -> None:
    if using_postgres():
        _init_postgres_database()
        return
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                headline TEXT,
                location TEXT,
                skills TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recommendation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                target_role TEXT,
                provider TEXT,
                input_payload TEXT NOT NULL,
                output_payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS saved_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_title TEXT NOT NULL,
                company TEXT NOT NULL,
                source TEXT,
                match_score INTEGER,
                reason TEXT,
                apply_link TEXT,
                job_payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, apply_link),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )
        _ensure_sqlite_columns(connection)


def _ensure_sqlite_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    for column_name, column_type in {
        "headline": "TEXT",
        "location": "TEXT",
        "skills": "TEXT",
    }.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}")


def _init_postgres_database() -> None:
    with connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT,
                    password_hash TEXT NOT NULL,
                    headline TEXT,
                    location TEXT,
                    skills TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_runs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    target_role TEXT,
                    provider TEXT,
                    input_payload TEXT NOT NULL,
                    output_payload TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_jobs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    job_title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    source TEXT,
                    match_score INTEGER,
                    reason TEXT,
                    apply_link TEXT,
                    job_payload TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, apply_link)
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at BIGINT NOT NULL,
                    used_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            for column_name in ("headline", "location", "skills"):
                cursor.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {column_name} TEXT")


def create_user(email: str, password: str, full_name: str | None = None) -> dict[str, Any]:
    normalized_email = email.strip().lower()
    if using_postgres():
        try:
            user_id = _postgres_fetch_value(
                """
                INSERT INTO users (email, full_name, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (normalized_email, full_name, hash_password(password)),
            )
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise ValueError("User already exists.") from exc
            raise
        return get_user_by_id(int(user_id)) or {}

    with connect() as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users (email, full_name, password_hash) VALUES (?, ?, ?)",
                (normalized_email, full_name, hash_password(password)),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("User already exists.") from exc
        user_id = int(cursor.lastrowid)
    return get_user_by_id(user_id) or {}


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, str(user["password_hash"])):
        return None
    return user


def get_user_by_email(email: str) -> dict[str, Any] | None:
    if using_postgres():
        return _postgres_fetch_one(
            "SELECT * FROM users WHERE email = %s",
            (email.strip().lower(),),
        )
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    if using_postgres():
        return _postgres_fetch_one("SELECT * FROM users WHERE id = %s", (user_id,))
    with connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def update_user_profile(
    user_id: int,
    *,
    full_name: str | None = None,
    headline: str | None = None,
    location: str | None = None,
    skills: str | None = None,
) -> dict[str, Any]:
    values = {
        "full_name": full_name,
        "headline": headline,
        "location": location,
        "skills": skills,
    }
    if using_postgres():
        _postgres_fetch_one(
            """
            UPDATE users
            SET full_name = %s, headline = %s, location = %s, skills = %s
            WHERE id = %s
            RETURNING id
            """,
            (*values.values(), user_id),
        )
        return get_user_by_id(user_id) or {}

    with connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET full_name = ?, headline = ?, location = ?, skills = ?
            WHERE id = ?
            """,
            (*values.values(), user_id),
        )
    return get_user_by_id(user_id) or {}


def update_user_password(user_id: int, password: str) -> None:
    if using_postgres():
        _postgres_execute_rowcount(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(password), user_id),
        )
        return
    with connect() as connection:
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user_id),
        )


def save_recommendation_run(
    *,
    user_id: int | None,
    target_role: str,
    provider: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
) -> int:
    if using_postgres():
        run_id = _postgres_fetch_value(
            """
            INSERT INTO recommendation_runs
                (user_id, target_role, provider, input_payload, output_payload)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                target_role,
                provider,
                json.dumps(input_payload),
                json.dumps(output_payload),
            ),
        )
        return int(run_id)

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO recommendation_runs
                (user_id, target_role, provider, input_payload, output_payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                target_role,
                provider,
                json.dumps(input_payload),
                json.dumps(output_payload),
            ),
        )
        return int(cursor.lastrowid)


def list_recommendation_runs(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    if using_postgres():
        rows = _postgres_fetch_all(
            """
            SELECT id, target_role, provider, output_payload, created_at
            FROM recommendation_runs
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return _deserialize_history(rows)

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, target_role, provider, output_payload, created_at
            FROM recommendation_runs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return _deserialize_history(rows)


def save_job(user_id: int, job: dict[str, Any]) -> int:
    apply_link = str(job.get("apply_link") or "")
    params = (
        user_id,
        str(job.get("job_title") or ""),
        str(job.get("company") or ""),
        str(job.get("source") or ""),
        int(job.get("match_score") or 0),
        str(job.get("reason") or ""),
        apply_link,
        json.dumps(job),
    )
    if using_postgres():
        saved_job_id = _postgres_fetch_value(
            """
            INSERT INTO saved_jobs
                (user_id, job_title, company, source, match_score, reason, apply_link, job_payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, apply_link) DO UPDATE SET
                job_title = EXCLUDED.job_title,
                company = EXCLUDED.company,
                source = EXCLUDED.source,
                match_score = EXCLUDED.match_score,
                reason = EXCLUDED.reason,
                job_payload = EXCLUDED.job_payload
            RETURNING id
            """,
            params,
        )
        return int(saved_job_id)

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR REPLACE INTO saved_jobs
                (user_id, job_title, company, source, match_score, reason, apply_link, job_payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        return int(cursor.lastrowid)


def list_saved_jobs(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    if using_postgres():
        rows = _postgres_fetch_all(
            """
            SELECT id, job_payload, created_at
            FROM saved_jobs
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return _deserialize_saved_jobs(rows)

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, job_payload, created_at
            FROM saved_jobs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return _deserialize_saved_jobs(rows)


def delete_saved_job(user_id: int, saved_job_id: int) -> bool:
    if using_postgres():
        return _postgres_execute_rowcount(
            "DELETE FROM saved_jobs WHERE id = %s AND user_id = %s",
            (saved_job_id, user_id),
        ) > 0

    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM saved_jobs WHERE id = ? AND user_id = ?",
            (saved_job_id, user_id),
        )
        return cursor.rowcount > 0


def create_password_reset_token(email: str, token: str, ttl_seconds: int = 900) -> dict[str, Any] | None:
    user = get_user_by_email(email)
    if not user:
        return None
    expires_at = int(time.time()) + ttl_seconds
    if using_postgres():
        _postgres_execute_rowcount(
            """
            INSERT INTO password_reset_tokens (token, user_id, expires_at)
            VALUES (%s, %s, %s)
            """,
            (token, int(user["id"]), expires_at),
        )
        return {"token": token, "expires_at": expires_at, "user": user}
    with connect() as connection:
        connection.execute(
            "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, int(user["id"]), expires_at),
        )
    return {"token": token, "expires_at": expires_at, "user": user}


def consume_password_reset_token(token: str, new_password: str) -> bool:
    now = int(time.time())
    if using_postgres():
        row = _postgres_fetch_one(
            """
            SELECT token, user_id, expires_at, used_at
            FROM password_reset_tokens
            WHERE token = %s
            """,
            (token,),
        )
        if not row or row.get("used_at") or int(row["expires_at"]) < now:
            return False
        update_user_password(int(row["user_id"]), new_password)
        _postgres_execute_rowcount(
            "UPDATE password_reset_tokens SET used_at = CURRENT_TIMESTAMP WHERE token = %s",
            (token,),
        )
        return True

    with connect() as connection:
        row = connection.execute(
            """
            SELECT token, user_id, expires_at, used_at
            FROM password_reset_tokens
            WHERE token = ?
            """,
            (token,),
        ).fetchone()
        if not row or row["used_at"] or int(row["expires_at"]) < now:
            return False
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), int(row["user_id"])),
        )
        connection.execute(
            "UPDATE password_reset_tokens SET used_at = CURRENT_TIMESTAMP WHERE token = ?",
            (token,),
        )
    return True


def admin_stats() -> dict[str, int]:
    if using_postgres():
        rows = _postgres_fetch_all(
            """
            SELECT 'users' AS name, COUNT(*) AS count FROM users
            UNION ALL
            SELECT 'recommendation_runs' AS name, COUNT(*) AS count FROM recommendation_runs
            UNION ALL
            SELECT 'saved_jobs' AS name, COUNT(*) AS count FROM saved_jobs
            """,
            (),
        )
        return {str(row["name"]): int(row["count"]) for row in rows}

    with connect() as connection:
        return {
            "users": int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]),
            "recommendation_runs": int(
                connection.execute("SELECT COUNT(*) FROM recommendation_runs").fetchone()[0]
            ),
            "saved_jobs": int(connection.execute("SELECT COUNT(*) FROM saved_jobs").fetchone()[0]),
        }


def _postgres_fetch_one(query: str, params: Iterable[Any]) -> dict[str, Any] | None:
    with connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()
    return dict(row) if row else None


def _postgres_fetch_all(query: str, params: Iterable[Any]) -> list[dict[str, Any]]:
    with connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
    return [dict(row) for row in rows]


def _postgres_fetch_value(query: str, params: Iterable[Any]) -> Any:
    row = _postgres_fetch_one(query, params)
    if not row:
        raise RuntimeError("PostgreSQL query did not return a value.")
    return next(iter(row.values()))


def _postgres_execute_rowcount(query: str, params: Iterable[Any]) -> int:
    with connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return int(cursor.rowcount)


def _deserialize_history(rows: Iterable[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["created_at"] = str(item.get("created_at") or "")
        item["output_payload"] = json.loads(item["output_payload"])
        history.append(item)
    return history


def _deserialize_saved_jobs(rows: Iterable[Any]) -> list[dict[str, Any]]:
    saved_jobs: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        job_payload = json.loads(item["job_payload"])
        job_payload["saved_job_id"] = item["id"]
        job_payload["saved_at"] = str(item.get("created_at") or "")
        saved_jobs.append(job_payload)
    return saved_jobs
