import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from model import SentimentModel

load_dotenv()

app = Flask(__name__)

# Load once at startup — shared across all requests
_model = SentimentModel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message: str, status: int):
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Body : JSON array of tweet strings  →  ["tweet1", "tweet2", ...]
    Response: {"tweet1": score1, "tweet2": score2, ...}
    Scores are floats in [-1.0, 1.0].
    """
    if not request.is_json:
        return _error("Content-Type must be application/json", 400)

    data = request.get_json(silent=True)
    if data is None:
        return _error("Request body is not valid JSON", 400)

    if not isinstance(data, list):
        return _error("Request body must be a JSON array of strings", 400)

    if len(data) == 0:
        return _error("Tweet list cannot be empty", 400)

    for i, item in enumerate(data):
        if not isinstance(item, str):
            return _error(f"Element at index {i} must be a string, got {type(item).__name__}", 400)
        if item.strip() == "":
            return _error(f"Tweet at index {i} is empty or contains only whitespace", 400)

    results = {tweet: round(_model.predict_score(tweet), 4) for tweet in data}
    return jsonify(results), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Generic error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("Method not allowed", 405)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
