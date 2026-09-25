#!/usr/bin/env python3
"""
check_embedding_progress.py
===========================
Checks status, processed article count, throughput, and sample records
from data/news_bge_m3_embeddings.db.
"""

import sys
import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path("/home/talha/projects/brazil-energy/data/news_bge_m3_embeddings.db")
TOTAL_ARTICLES = 26937

if not DB_PATH.exists():
    print(f"Database {DB_PATH} not found yet.")
    sys.exit(0)

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM articles")
completed = cursor.fetchone()[0]

cursor.execute("SELECT MIN(processed_at), MAX(processed_at) FROM articles")
min_t, max_t = cursor.fetchone()

pct = (completed / TOTAL_ARTICLES) * 100 if TOTAL_ARTICLES > 0 else 0

print("=" * 60)
print("BGE-M3 News Embedding Progress")
print("=" * 60)
print(f"Completed: {completed:,} / {TOTAL_ARTICLES:,} articles ({pct:.2f}%)")
print(f"First processed: {min_t}")
print(f"Last processed : {max_t}")

cursor.execute("""
    SELECT filename, date, region, topic, title, word_count, embedding 
    FROM articles 
    ORDER BY id DESC LIMIT 1
""")
latest = cursor.fetchone()
if latest:
    emb = np.frombuffer(latest[6], dtype=np.float32)
    print("\n--- Latest Processed Article ---")
    print(f"File   : {latest[0]}")
    print(f"Date   : {latest[1]} | Region: {latest[2]} | Topic: {latest[3]}")
    print(f"Title  : {latest[4]}")
    print(f"Words  : {latest[5]}")
    print(f"Vector : shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

conn.close()
print("=" * 60)
