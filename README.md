# SocialMetrics AI — Sentiment Analysis API

Flask API for sentiment analysis of tweets, using Logistic Regression on TF-IDF features.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

### 3. Initialise the database
```bash
mysql -u root -p < setup_db.sql
```

### 4. Run the API
```bash
python app.py
```

The server starts on `http://localhost:5000` (override with `PORT` env var).

---

## API Reference

### `POST /analyze`

Analyse the sentiment of one or more tweets.

**Request**
```
Content-Type: application/json

["tweet text 1", "tweet text 2", ...]
```

**Response `200`**
```json
{
  "tweet text 1": 0.82,
  "tweet text 2": -0.45
}
```
Scores are in **[−1.0, 1.0]**: −1 = very negative, 0 = neutral, 1 = very positive.

**Error responses**

| Status | Cause |
|--------|-------|
| 400 | Body is not JSON / not an array / empty array / non-string element / empty tweet |
| 404 | Unknown endpoint |
| 405 | Wrong HTTP method |

**Examples**
```bash
# Analyse two tweets
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d '["I love this!", "This is terrible"]'

# Empty list → 400
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d '[]'
```

### `GET /health`
Returns `{"status": "ok"}` — useful for uptime monitoring.

---

## Model retraining

Run manually:
```bash
python retrain.py
```

The script reads all rows from the `tweets` table and retrains both classifiers.  
Requires at least **20** labelled samples.

**Weekly cron (Linux/macOS)**
```
0 2 * * 0 /usr/bin/python3 /path/to/retrain.py >> /var/log/retrain.log 2>&1
```

**Windows Task Scheduler**
- Program: `python`
- Arguments: `C:\path\to\retrain.py`
- Trigger: Weekly, Sunday, 02:00

---

## Project structure

```
├── app.py          # Flask API
├── model.py        # TF-IDF + LogisticRegression model
├── db.py           # MySQL connection & queries
├── retrain.py      # Retraining script
├── setup_db.sql    # DB initialisation + seed data
├── requirements.txt
└── .env.example
```
