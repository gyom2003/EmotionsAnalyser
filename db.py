import os
from pathlib import Path
import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_SAMPLE_TWEETS = [
    ("I love this product, it's amazing!", 1, 0),
    ("Great service, very happy with the results", 1, 0),
    ("This is the best thing ever, highly recommend", 1, 0),
    ("Absolutely fantastic experience, will definitely return", 1, 0),
    ("So happy today, everything is going wonderfully", 1, 0),
    ("Excellent work, very impressed with the quality", 1, 0),
    ("This is terrible, worst experience I ever had", 0, 1),
    ("Very disappointed, complete waste of money and time", 0, 1),
    ("Awful product, broken on arrival, do not buy", 0, 1),
    ("Horrible service, never going back to this place", 0, 1),
    ("The product arrived today in its original packaging", 0, 0),
    ("Here is my review of the product after testing it", 0, 0),
]


def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "socialmetrics"),
    )


def get_admin_connection():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        autocommit=True,
    )


def initialize_database() -> None:
    """Create the database and seed sample tweets if the table is empty."""
    db_name = os.environ.get("DB_NAME", "socialmetrics")
    conn = get_admin_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute(f"USE `{db_name}`")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                positive TINYINT(1) NOT NULL DEFAULT 0,
                negative TINYINT(1) NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("SELECT COUNT(*) FROM tweets")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
                DEFAULT_SAMPLE_TWEETS,
            )
            conn.commit()
    finally:
        conn.close()


def fetch_training_data() -> tuple[list[str], list[int], list[int]]:
    """Return (texts, positive_labels, negative_labels) from the tweets table."""
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT text, positive, negative FROM tweets")
        rows = cursor.fetchall()
    finally:
        conn.close()

    texts = [row["text"] for row in rows]
    positive_labels = [int(row["positive"]) for row in rows]
    negative_labels = [int(row["negative"]) for row in rows]
    return texts, positive_labels, negative_labels


def insert_tweet(text: str, positive: int = 0, negative: int = 0) -> int:
    """Insert a tweet record and return the new id."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)",
            (text, int(positive), int(negative)),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_tweets(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return a list of tweets (dicts)."""
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, text, positive, negative, created_at FROM tweets ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def get_tweet(tweet_id: int) -> dict | None:
    """Return a single tweet by id or None if not found."""
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, text, positive, negative, created_at FROM tweets WHERE id = %s",
            (tweet_id,),
        )
        return cursor.fetchone()
    finally:
        conn.close()
