"""
Generates the model evaluation report (confusion matrices + precision/recall/F1)
for the `positive` and `negative` classifiers.

Data source priority:
  1. The live `tweets` table in MySQL (via db.fetch_training_data()), so the
     report always reflects whatever has been annotated so far.
  2. If no database is reachable, falls back to parsing the seed dataset
     shipped in ../setup_db.sql, so the report can still be regenerated
     offline / in CI.

Because the annotated dataset is small, a single train/test split would
leave very few validation examples. Instead we use 5-fold stratified
cross-validation and aggregate the out-of-fold predictions with
cross_val_predict, which yields one prediction per tweet while every
prediction is still made by a model that never saw that tweet during
training — the standard approach for evaluating on small datasets.

Usage:
    python evaluate.py
Outputs (written to this directory):
    confusion_matrix_positive.png
    confusion_matrix_negative.png
    evaluation_report.pdf
"""
import os
import re
import sys
import numpy as np

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SQL_SEED_FILE = os.path.join(ROOT, "setup_db.sql")
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_sql_seed(path: str):
    """Parse the INSERT INTO tweets (...) VALUES (...) block of setup_db.sql."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r"INSERT INTO tweets.*?VALUES\s*(.*?);", content, re.S)
    if not match:
        raise ValueError(f"Could not find an INSERT INTO tweets statement in {path}")

    rows_blob = match.group(1)
    row_pattern = re.compile(
        r'\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*(\d)\s*,\s*(\d)\s*\)', re.S
    )
    texts, positive, negative = [], [], []
    for text, pos, neg in row_pattern.findall(rows_blob):
        texts.append(text.replace('\\"', '"'))
        positive.append(int(pos))
        negative.append(int(neg))

    if not texts:
        raise ValueError(f"No rows parsed from {path}")
    return texts, positive, negative


def load_training_data():
    """Try the live database first, fall back to the SQL seed file."""
    sys.path.insert(0, ROOT)
    try:
        from db import fetch_training_data
        texts, positive, negative = fetch_training_data()
        if len(texts) >= 10:
            print(f"[data] Loaded {len(texts)} annotated tweets from the MySQL `tweets` table.")
            return texts, positive, negative
        print("[data] Database reachable but fewer than 10 rows found, using SQL seed instead.")
    except Exception as exc:
        print(f"[data] Database unavailable ({exc.__class__.__name__}: {exc}). Using SQL seed instead.")

    texts, positive, negative = load_from_sql_seed(SQL_SEED_FILE)
    print(f"[data] Loaded {len(texts)} annotated tweets from setup_db.sql.")
    return texts, positive, negative


# ---------------------------------------------------------------------------
# Cross-validated evaluation for one binary label (positive or negative)
# ---------------------------------------------------------------------------

def evaluate_label(texts, labels, label_name: str, n_folds: int = N_FOLDS):
    labels = np.array(labels)
    n_folds = min(n_folds, int(np.bincount(labels).min())) if len(np.unique(labels)) > 1 else 2
    n_folds = max(n_folds, 2)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, labels, cv=skf)
    y_proba = cross_val_predict(clf, X, labels, cv=skf, method="predict_proba")[:, 1]

    cm = confusion_matrix(labels, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "label_name": label_name,
        "n_folds": n_folds,
        "confusion_matrix": cm,
        "precision": precision,   # [class 0, class 1]
        "recall": recall,
        "f1": f1,
        "support": support,
        "y_true": labels,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, label_name: str, class_names=("Non", "Oui")):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matrice de confusion — classe « {label_name} »")
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Annotation réelle")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bias / error analysis (data-driven, computed from the actual results)
# ---------------------------------------------------------------------------

def analyze_errors(texts, pos_result, neg_result):
    notes = []

    for result, name in ((pos_result, "positive"), (neg_result, "negative")):
        cm = result["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        support_pos_class = result["support"][1]
        total = cm.sum()
        proba = result["y_proba"]
        recall_pos = result["recall"][1]

        if recall_pos == 0.0 and tp == 0:
            spread = proba.max() - proba.min()
            notes.append(
                f"- Biais majeur observé sur la classe « {name} » : en validation croisée "
                f"(donc sur des tweets que le modèle n'a jamais vus à l'entraînement), aucune "
                f"prédiction ne dépasse le seuil de 0.5 (probabilités comprises entre "
                f"{proba.min():.2f} et {proba.max():.2f}, écart de seulement {spread:.2f}). "
                f"Le modèle ne fait donc que reconnaître les tweets déjà mémorisés pendant "
                f"l'entraînement et échoue à généraliser à de nouveaux tweets pour cette classe."
            )
            notes.append(
                f"- Cause probable : avec seulement {total} tweets annotés et un vectoriseur "
                f"TF-IDF autorisant jusqu'à 5000 features en unigrammes+bigrammes, l'espace des "
                f"caractéristiques est bien plus grand que le nombre d'exemples. Chaque pli de "
                f"validation entraîne le modèle sur un vocabulaire trop spécifique aux tweets vus, "
                f"qui ne se retrouve pas forcément dans les tweets laissés de côté."
            )
        elif support_pos_class / total < 0.35:
            notes.append(
                f"- La classe « {name} » ne représente que {support_pos_class}/{total} tweets "
                f"({support_pos_class/total:.0%}) du jeu annoté : le modèle voit peu d'exemples "
                f"positifs pour cette étiquette, ce qui l'expose à un biais vers la classe majoritaire."
            )
        if fn > fp and tp > 0:
            notes.append(
                f"- Pour la classe « {name} », le modèle rate plus de cas réels qu'il n'en "
                f"invente ({fn} faux négatifs contre {fp} faux positifs) : il a tendance à "
                f"sous-détecter ce sentiment."
            )
        elif fp > fn:
            notes.append(
                f"- Pour la classe « {name} », le modèle sur-détecte ({fp} faux positifs contre "
                f"{fn} faux négatifs) : il classe parfois des tweets neutres ou de l'autre "
                f"sentiment comme « {name} »."
            )

    lengths = [len(t.split()) for t in texts]
    notes.append(
        f"- Les tweets du jeu de données sont courts (en moyenne {np.mean(lengths):.1f} mots), "
        f"ce qui limite le nombre de signaux lexicaux disponibles pour le TF-IDF et augmente "
        f"le risque de confusion sur des formulations inhabituelles ou ironiques."
    )
    notes.append(
        "- Le jeu de données mélange français et anglais avec un vocabulaire globalement "
        "explicite (« terrible », « fantastique », « nul »...) ; des tweets plus subtils, "
        "sarcastiques ou implicites seraient probablement mal classés par ce vectoriseur "
        "purement lexical (TF-IDF), qui ne capture pas le contexte ou la négation à longue distance."
    )
    return notes


RECOMMENDATIONS = [
    "Priorité n°1 : augmenter significativement la taille du jeu de données annoté. Avec "
    "seulement ~27 tweets, le modèle ne parvient à prédire aucun cas positif hors échantillon "
    "(voir l'analyse des biais) — il faut au moins plusieurs centaines de tweets par classe "
    "avant d'attendre des performances fiables.",
    "Réduire la dimensionnalité du TF-IDF pour ce volume de données (par ex. max_features réduit "
    "à quelques centaines, ou ngram_range=(1,1) le temps que le corpus grandisse) afin de limiter "
    "le sur-apprentissage sur un vocabulaire trop spécifique aux tweets d'entraînement.",
    "Ajuster la régularisation (paramètre C de LogisticRegression) par validation croisée plutôt "
    "que de garder C=1.0 par défaut, pour trouver le compromis biais/variance adapté à la taille "
    "réelle du jeu de données.",
    "Ajouter une vraie classe « neutre » explicite dans la table `tweets` (aujourd'hui encodée "
    "implicitement par positive=0 ET negative=0) pour éviter que le modèle ne confonde neutre et "
    "sentiment faible.",
    "À terme, remplacer ou compléter le TF-IDF par des embeddings contextuels (ex. "
    "sentence-transformers) pour mieux capturer la négation, le sarcasme et le contexte au-delà "
    "du simple sac de mots.",
    "Suivre les métriques (précision/rappel/F1) à chaque réentraînement hebdomadaire et déclencher "
    "une alerte si le F1 d'une classe chute, afin de détecter une dérive du modèle (data drift).",
    "Mettre en place une revue humaine périodique d'un échantillon des tweets mal classés pour "
    "affiner les règles d'annotation et réduire les incohérences entre annotateurs.",
]


# ---------------------------------------------------------------------------
# PDF report assembly
# ---------------------------------------------------------------------------

def render_text_page(pdf, title, lines, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    y = 0.88
    has_content = True
    for i, line in enumerate(lines):
        fig.text(0.08, y, line, fontsize=fontsize, va="top", wrap=True)
        n_wraps = max(1, len(line) // 95 + 1)
        y -= 0.032 * n_wraps + 0.008
        if y < 0.05 and i < len(lines) - 1:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.93
            has_content = False
        else:
            has_content = True
    if has_content:
        pdf.savefig(fig)
    plt.close(fig)


def render_metrics_table(result, class_labels):
    fig, ax = plt.subplots(figsize=(8.27, 3))
    ax.axis("off")
    ax.set_title(f"Précision / Rappel / F1-score — classe « {result['label_name']} »",
                 fontsize=13, fontweight="bold", pad=20)

    rows = []
    for idx, cls in enumerate(class_labels):
        rows.append([
            cls,
            f"{result['precision'][idx]:.2f}",
            f"{result['recall'][idx]:.2f}",
            f"{result['f1'][idx]:.2f}",
            f"{result['support'][idx]}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Classe", "Précision", "Rappel", "F1-score", "Support"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    fig.tight_layout()
    return fig


def build_pdf(pos_result, neg_result, bias_notes, n_samples):
    pdf_path = os.path.join(HERE, "evaluation_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # --- Cover / methodology page ---------------------------------
        render_text_page(pdf, "SocialMetrics AI — Rapport d'évaluation du modèle", [
            "",
            "Analyse de sentiments des tweets — Régression logistique (TF-IDF)",
            "",
            f"Jeu de données annoté évalué : {n_samples} tweets (table `tweets`).",
            "",
            "Méthodologie :",
            "- Deux classifieurs binaires indépendants (comme en production) : un pour la",
            "  colonne `positive`, un pour la colonne `negative`.",
            f"- Évaluation par validation croisée stratifiée à {pos_result['n_folds']} plis (positive) "
            f"et {neg_result['n_folds']} plis (negative), avec cross_val_predict : chaque tweet",
            "  est prédit par un modèle qui ne l'a jamais vu à l'entraînement.",
            "- Ce choix évite de sacrifier une grande partie d'un jeu de données restreint",
            "  dans un simple split train/test, tout en gardant une évaluation honnête.",
            "",
            "Ce rapport présente, pour chacune des deux classes (positive et negative) :",
            "  1. La matrice de confusion des prédictions,",
            "  2. Les métriques de précision, rappel et F1-score,",
            "  3. Une analyse des biais et erreurs observés,",
            "  4. Des recommandations d'amélioration.",
        ])

        # --- Positive: confusion matrix + metrics ----------------------
        fig_cm_pos = plot_confusion_matrix(pos_result["confusion_matrix"], "positive")
        pdf.savefig(fig_cm_pos)
        plt.close(fig_cm_pos)
        fig_cm_pos.savefig(os.path.join(HERE, "confusion_matrix_positive.png"), dpi=150)

        tn, fp, fn, tp = pos_result["confusion_matrix"].ravel()
        render_text_page(pdf, "Interprétation — Matrice de confusion « positive »", [
            f"Vrais négatifs (VN) : {tn}   |   Faux positifs (FP) : {fp}",
            f"Faux négatifs (FN) : {fn}   |   Vrais positifs (VP) : {tp}",
            "",
            f"Le modèle identifie correctement {tp} des {tp+fn} tweets réellement positifs "
            f"(rappel classe positive = {pos_result['recall'][1]:.2f}), et parmi les tweets qu'il "
            f"annonce positifs, {tp}/{tp+fp} le sont vraiment "
            f"(précision classe positive = {pos_result['precision'][1]:.2f}).",
        ])

        fig_table_pos = render_metrics_table(pos_result, ["Non-positive (0)", "Positive (1)"])
        pdf.savefig(fig_table_pos)
        plt.close(fig_table_pos)

        # --- Negative: confusion matrix + metrics -----------------------
        fig_cm_neg = plot_confusion_matrix(neg_result["confusion_matrix"], "negative")
        pdf.savefig(fig_cm_neg)
        plt.close(fig_cm_neg)
        fig_cm_neg.savefig(os.path.join(HERE, "confusion_matrix_negative.png"), dpi=150)

        tn, fp, fn, tp = neg_result["confusion_matrix"].ravel()
        render_text_page(pdf, "Interprétation — Matrice de confusion « negative »", [
            f"Vrais négatifs (VN) : {tn}   |   Faux positifs (FP) : {fp}",
            f"Faux négatifs (FN) : {fn}   |   Vrais positifs (VP) : {tp}",
            "",
            f"Le modèle identifie correctement {tp} des {tp+fn} tweets réellement négatifs "
            f"(rappel classe negative = {neg_result['recall'][1]:.2f}), et parmi les tweets qu'il "
            f"annonce négatifs, {tp}/{tp+fp} le sont vraiment "
            f"(précision classe negative = {neg_result['precision'][1]:.2f}).",
        ])

        fig_table_neg = render_metrics_table(neg_result, ["Non-négative (0)", "Négative (1)"])
        pdf.savefig(fig_table_neg)
        plt.close(fig_table_neg)

        # --- Bias / error analysis --------------------------------------
        render_text_page(pdf, "Analyse des biais et des erreurs", bias_notes, fontsize=10.5)

        # --- Recommendations ---------------------------------------------
        render_text_page(
            pdf,
            "Recommandations d'amélioration",
            [f"{i+1}. {rec}" for i, rec in enumerate(RECOMMENDATIONS)],
            fontsize=10.5,
        )

    return pdf_path


def main():
    texts, positive, negative = load_training_data()

    pos_result = evaluate_label(texts, positive, "positive")
    neg_result = evaluate_label(texts, negative, "negative")

    print("\n=== Classe positive ===")
    print("Matrice de confusion [[VN, FP], [FN, VP]] :")
    print(pos_result["confusion_matrix"])
    print(f"Précision: {pos_result['precision']} | Rappel: {pos_result['recall']} | F1: {pos_result['f1']}")

    print("\n=== Classe negative ===")
    print("Matrice de confusion [[VN, FP], [FN, VP]] :")
    print(neg_result["confusion_matrix"])
    print(f"Précision: {neg_result['precision']} | Rappel: {neg_result['recall']} | F1: {neg_result['f1']}")

    bias_notes = analyze_errors(texts, pos_result, neg_result)

    pdf_path = build_pdf(pos_result, neg_result, bias_notes, len(texts))
    print(f"\nRapport PDF généré : {pdf_path}")


if __name__ == "__main__":
    main()
