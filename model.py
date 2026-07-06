import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

# Seed data used when no trained model file exists yet
_SEED_POSITIVE = [
    "I love this product, it's amazing!",
    "Great service, very happy with the results",
    "This is the best thing ever, highly recommend",
    "Absolutely fantastic experience, will definitely return",
    "So happy today, everything is going wonderfully",
    "Excellent work, very impressed with the quality",
    "Wonderful, this made my day so much better",
    "Super excited about this, outstanding performance",
    "J'adore ce produit, c'est vraiment fantastique",
    "Très satisfait du service, je recommande vivement",
    "Excellente expérience, merci beaucoup pour tout",
    "Super content, tout se passe parfaitement bien",
    "Bravo, c'est vraiment impressionnant et bien fait",
    "Perfect, exactly what I needed, thank you so much",
    "Incredible results, I am beyond pleased with this",
]

_SEED_NEGATIVE = [
    "This is terrible, worst experience I ever had",
    "Very disappointed, complete waste of money and time",
    "Awful product, broken on arrival, do not buy",
    "Horrible service, never going back to this place",
    "So frustrated, nothing works as advertised at all",
    "Worst decision I ever made, extremely bad quality",
    "Completely useless and defective, very angry",
    "This ruined my day, unacceptable and outrageous",
    "C'est nul, très déçu du service client",
    "Horrible expérience, je ne recommande absolument pas",
    "Très mauvais produit, total gâchis d'argent",
    "Complètement inutile et cassé dès le premier jour",
    "Furieux, rien ne fonctionne, service catastrophique",
    "Disgraceful quality, I want a full refund immediately",
    "Appalling, this is the worst purchase I have ever made",
]

_SEED_NEUTRAL = [
    "The product arrived today in its original packaging",
    "I ordered this item last week from the website",
    "The package was delivered as scheduled",
    "I have been using this for about a month now",
    "Here is my review of the product after testing it",
    "The item matches the description on the page",
    "Le produit est arrivé aujourd'hui dans son emballage",
    "J'ai commandé ça la semaine dernière sur le site",
    "Voici mon avis après quelques semaines d'utilisation",
    "The delivery took three days, which was expected",
]


class SentimentModel:
    """
    Two LogisticRegression classifiers (positive / negative) on top of TF-IDF.

    Score = P(positive=1) − P(negative=1), clamped to [−1, 1].
    """

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            try:
                self._load()
                return
            except Exception:
                pass

        try:
            from db import fetch_training_data

            texts, positive_labels, negative_labels = fetch_training_data()
            if len(texts) >= 8:
                self._train(texts, positive_labels, negative_labels)
                return
        except Exception:
            pass

        self._train(_SEED_POSITIVE + _SEED_NEGATIVE + _SEED_NEUTRAL,
                    [1] * len(_SEED_POSITIVE) + [0] * len(_SEED_NEGATIVE) + [0] * len(_SEED_NEUTRAL),
                    [0] * len(_SEED_POSITIVE) + [1] * len(_SEED_NEGATIVE) + [0] * len(_SEED_NEUTRAL))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_score(self, tweet: str) -> float:
        X = self.vectorizer.transform([tweet])
        p_pos = self.pos_model.predict_proba(X)[0][1]
        p_neg = self.neg_model.predict_proba(X)[0][1]
        return float(np.clip(p_pos - p_neg, -1.0, 1.0))

    def train(self, texts: list[str], positive_labels: list[int], negative_labels: list[int]):
        self._train(texts, positive_labels, negative_labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(self, texts, positive_labels, negative_labels):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)

        self.pos_model = LogisticRegression(max_iter=1000, C=1.0)
        self.pos_model.fit(X, positive_labels)

        self.neg_model = LogisticRegression(max_iter=1000, C=1.0)
        self.neg_model.fit(X, negative_labels)

        self._save()

    def _save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer,
                 "pos_model": self.pos_model,
                 "neg_model": self.neg_model},
                f,
            )

    def _load(self):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.pos_model = data["pos_model"]
        self.neg_model = data["neg_model"]
