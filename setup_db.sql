-- Run once to initialise the database and seed sample data.
-- Usage: mysql -u root -p < setup_db.sql

CREATE DATABASE IF NOT EXISTS socialmetrics
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE socialmetrics;

CREATE TABLE IF NOT EXISTS tweets (
  id         INT AUTO_INCREMENT PRIMARY KEY,
  text       TEXT        NOT NULL,
  positive   TINYINT(1)  NOT NULL DEFAULT 0,
  negative   TINYINT(1)  NOT NULL DEFAULT 0,
  created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

-- ---- Sample data -------------------------------------------------------
INSERT INTO tweets (text, positive, negative) VALUES
  ("I love this product, it's amazing!", 1, 0),
  ("Great service, very happy with the results", 1, 0),
  ("Absolutely fantastic, highly recommend", 1, 0),
  ("So happy today, everything is going well", 1, 0),
  ("Excellent work, very impressed", 1, 0),
  ("Wonderful experience, will come back", 1, 0),
  ("This made my day so much better", 1, 0),
  ("Outstanding performance, well done", 1, 0),
  ("J'adore ce produit, c'est fantastique!", 1, 0),
  ("Très satisfait du service, je recommande", 1, 0),
  ("Excellente expérience, merci beaucoup", 1, 0),
  ("Super content, tout se passe bien", 1, 0),
  ("This is terrible, worst experience ever", 0, 1),
  ("Very disappointed, complete waste of money", 0, 1),
  ("Awful product, do not recommend", 0, 1),
  ("Horrible service, never going back", 0, 1),
  ("So frustrated with this, nothing works", 0, 1),
  ("Worst decision I ever made", 0, 1),
  ("Completely useless and broken", 0, 1),
  ("Unacceptable quality, very bad", 0, 1),
  ("C'est nul, très déçu du service", 0, 1),
  ("Horrible expérience, je ne recommande pas", 0, 1),
  ("Très mauvais produit, total gâchis", 0, 1),
  ("The product arrived today", 0, 0),
  ("I ordered this last week", 0, 0),
  ("The package was delivered", 0, 0),
  ("Here is my review of the product", 0, 0);
