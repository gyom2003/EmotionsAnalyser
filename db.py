import os
import mysql.connector
from mysql.connector import Error


def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 3306)),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "socialmetrics"),
    )


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
