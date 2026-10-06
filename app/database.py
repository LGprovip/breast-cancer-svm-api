import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


def create_user(username: str, email: str, password_hash: str):
    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO users
            (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, email, password_hash)
        )

        connection.commit()

        return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None

    finally:
        connection.close()


def get_user_by_username(username: str):
    connection = get_connection()

    user = connection.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        """,
        (username,)
    ).fetchone()

    connection.close()

    return user