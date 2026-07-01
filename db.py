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
