import mysql.connector

conn = mysql.connector.connect(host='localhost', port=3306, user='root', password='', autocommit=True)
cur = conn.cursor()
cur.execute('CREATE DATABASE IF NOT EXISTS socialmetrics CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
cur.execute('USE socialmetrics')
cur.execute('CREATE TABLE IF NOT EXISTS tweets (id INT AUTO_INCREMENT PRIMARY KEY, text TEXT NOT NULL, positive TINYINT(1) NOT NULL DEFAULT 0, negative TINYINT(1) NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
rows = [
    ("J'adore ce produit", 1, 0),
    ("C'est horrible", 0, 1),
    ("Très satisfait du service", 1, 0),
]
cur.executemany('INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)', rows)
conn.commit()
cur.execute('SELECT COUNT(*) FROM tweets')
print(cur.fetchone()[0])
conn.close()
