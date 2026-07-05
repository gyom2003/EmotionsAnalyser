"""Examples for using the SocialMetrics API (Python `requests`).

Run with an active API server (default http://localhost:5000).
Requires: `pip install requests`
"""

import requests
from pprint import pprint

BASE = "http://localhost:5000"


def create_tweet(text: str, positive: int = 0, negative: int = 0):
    url = f"{BASE}/tweets"
    payload = {"text": text, "positive": positive, "negative": negative}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    return r.json()


def list_tweets(limit: int = 100, offset: int = 0):
    url = f"{BASE}/tweets"
    r = requests.get(url, params={"limit": limit, "offset": offset})
    r.raise_for_status()
    return r.json()


def get_tweet(tweet_id: int):
    url = f"{BASE}/tweets/{tweet_id}"
    r = requests.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def analyze(tweets: list[str]):
    url = f"{BASE}/analyze"
    r = requests.post(url, json=tweets)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print("Creating a positive tweet...")
    new = create_tweet("J'adore ce service, très satisfait !", positive=1, negative=0)
    pprint(new)

    print("\nListing tweets (limit=5)...")
    rows = list_tweets(limit=5)
    pprint(rows)

    if rows:
        tid = rows[0]["id"]
        print(f"\nGet tweet {tid}...")
        pprint(get_tweet(tid))

    print("\nAnalyze sample tweets...")
    res = analyze(["Très bon produit", "C'est catastrophique"])
    pprint(res)
