# Documentation de l'API — SocialMetrics AI

Cette documentation décrit l'installation, la configuration et les exemples de requêtes pour l'API.

## Installation

- Créer un environnement virtuel et installer les dépendances :

```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

- Copier le fichier d'exemple d'environnement et le modifier :

```bash
cp .env.example .env
# Éditer .env pour renseigner DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
```

## Initialiser la base de données MySQL

Exécuter :

```bash
mysql -u root -p < setup_db.sql
```

Cela crée la base `socialmetrics` et la table `tweets` (colonnes `id`, `text`, `positive`, `negative`, `created_at`).

## Endpoints

- POST /analyze
  - Body: JSON array de tweets (strings)
  - Response 200: JSON map tweet → score (float dans [-1.0, 1.0])

- GET /health
  - Response 200: `{"status": "ok"}`

- POST /tweets
  - Crée un tweet annoté (sert de dataset pour l'entraînement)
  - Body: JSON object

```json
{
  "text": "J'aime ce produit !",
  "positive": 1,
  "negative": 0
}
```

  - Response 201: `{ "id": <int>, "text": ..., "positive": 0|1, "negative": 0|1 }`
  - Erreurs 400 en cas de corps invalide ou champs manquants / mal typés.

- GET /tweets
  - Liste les tweets annotés.
  - Query params (optionnels): `limit` (défaut 100), `offset` (défaut 0)
  - Response 200: JSON array d'objets `{ "id", "text", "positive", "negative", "created_at" }`

- GET /tweets/{id}
  - Récupère un tweet par son `id`.
  - Response 200: objet tweet ou 404 si non trouvé.

## Exemples curl

- Ajouter un tweet :

```bash
curl -X POST http://localhost:5000/tweets \
  -H "Content-Type: application/json" \
  -d '{"text":"Très satisfait du service","positive":1,"negative":0}'
```

- Lister les tweets (50 premiers) :

```bash
curl "http://localhost:5000/tweets?limit=50&offset=0"
```

- Récupérer un tweet :

```bash
curl http://localhost:5000/tweets/1
```

- Analyser des tweets :

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '["J\'adore ce produit","C\'est horrible"]'
```

## Réponses attendues (exemples)

- `POST /tweets` → 201

```json
{ "id": 42, "text": "Très satisfait du service", "positive": 1, "negative": 0 }
```

- `GET /tweets` → 200

```json
[
  { "id": 42, "text": "Très satisfait du service", "positive": 1, "negative": 0, "created_at": "2026-07-05T12:00:00" },
  { "id": 41, "text": "Très déçu", "positive": 0, "negative": 1, "created_at": "2026-07-04T10:00:00" }
]
```

---

Si vous voulez, je peux aussi ajouter des exemples en Python (requests) ou un postman collection.
