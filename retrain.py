"""
Run this script to retrain the model from the tweets table.
Can be scheduled weekly via cron or Task Scheduler.

  Unix cron (every Sunday at 2 AM):
    0 2 * * 0 /usr/bin/python3 /path/to/retrain.py >> /var/log/retrain.log 2>&1

  Windows Task Scheduler:
    Action: python C:\path\to\retrain.py
    Trigger: Weekly, Sunday, 02:00
"""
import logging
from dotenv import load_dotenv
from db import fetch_training_data
from model import SentimentModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

MIN_SAMPLES = 20


def retrain():
    logger.info("Fetching training data from database...")
    texts, positive_labels, negative_labels = fetch_training_data()

    if len(texts) < MIN_SAMPLES:
        logger.warning(
            "Only %d samples available (minimum %d). Skipping retraining.",
            len(texts), MIN_SAMPLES,
        )
        return

    logger.info("Retraining on %d samples...", len(texts))
    model = SentimentModel.__new__(SentimentModel)
    model.train(texts, positive_labels, negative_labels)
    logger.info("Model retrained and saved successfully.")


if __name__ == "__main__":
    retrain()
