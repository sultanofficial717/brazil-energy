#!/usr/bin/env python3
"""
embed_news_articles.py
======================
Generates dense semantic embeddings (BAAI/BGE-M3, 1024 dimensions) for news articles
stored in archive.zip.

Crucial User Requirements:
1. Skips lines 1-5 (metadata: URL, Region, Date, Topic, empty separator).
2. ONLY embeds the news headline and body from line 6 onwards.
3. Preserves metadata in the database alongside the generated embeddings.
4. Resumable via SQLite checkpointing (skips already processed articles).
5. Exports:
   - SQLite database: data/news_bge_m3_embeddings.db
   - High-speed compressed NumPy archive: data/news_bge_m3_embeddings.npz
   - Consolidated article CSV: data/news_bge_m3_embeddings.csv
   - Date/Region aggregated embeddings: data/news_embeddings_by_date_region.csv
"""

import os
import sys
import time
import json
import zipfile
import sqlite3
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

DEFAULT_ZIP_PATH = Path("/home/talha/projects/brazil-energy/archive.zip")
DEFAULT_DB_PATH = Path("/home/talha/projects/brazil-energy/data/news_bge_m3_embeddings.db")
DEFAULT_NPZ_PATH = Path("/home/talha/projects/brazil-energy/data/news_bge_m3_embeddings.npz")
DEFAULT_CSV_PATH = Path("/home/talha/projects/brazil-energy/data/news_bge_m3_embeddings.csv")
DEFAULT_AGG_PATH = Path("/home/talha/projects/brazil-energy/data/news_embeddings_by_date_region.csv")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "bge-m3"

# Metadata / non-article files to ignore
IGNORE_FILES = {
    "About Data.txt",
    "verification_audit_report/About Data.txt",
    "verification_audit_report/verification_audit_report.txt",
    "verification_audit_report/filter_and_dedup_report.txt",
}


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database for storing embeddings and metadata."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            url TEXT,
            region TEXT,
            date TEXT,
            topic TEXT,
            title TEXT,
            word_count INTEGER,
            char_count INTEGER,
            embedding BLOB,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON articles(filename)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_region ON articles(date, region)")
    conn.commit()
    return conn


def get_completed_filenames(conn: sqlite3.Connection) -> set:
    """Retrieve set of already processed filenames."""
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM articles")
    return set(row[0] for row in cursor.fetchall())


def parse_article(filename: str, raw_content: str) -> Optional[Dict[str, Any]]:
    """
    Parses a news file according to the dataset structure:
    Line 1: URL
    Line 2: Region
    Line 3: Date
    Line 4: Topic
    Line 5: (Blank separator)
    Line 6+: News Headline + Body

    Returns None if the file is invalid or has no news body.
    """
    lines = raw_content.splitlines()
    if len(lines) < 6:
        # Fewer than 6 lines means no news body
        return None

    # Lines 1-5: Metadata
    url = lines[0].strip()
    region = lines[1].strip() if len(lines) > 1 else ""
    date = lines[2].strip() if len(lines) > 2 else ""
    raw_topic = lines[3].strip() if len(lines) > 3 else ""
    topic = raw_topic.replace("Topic:", "").strip() if raw_topic.startswith("Topic:") else raw_topic

    # Line 6+: News content (ONLY embed this part)
    # line 6 is index 5 in 0-indexed list
    news_lines = lines[5:]
    news_body = "\n".join(news_lines).strip()

    if not news_body:
        return None

    # First non-empty line of news body as title
    title = ""
    for nl in news_lines:
        s = nl.strip()
        if s:
            title = s
            break

    words = len(news_body.split())
    chars = len(news_body)

    return {
        "filename": filename,
        "url": url,
        "region": region,
        "date": date,
        "topic": topic,
        "title": title,
        "news_body": news_body,
        "word_count": words,
        "char_count": chars,
    }


def fetch_embeddings_batch(
    texts: List[str], model: str = MODEL_NAME, url: str = OLLAMA_EMBED_URL, timeout: int = 180
) -> List[List[float]]:
    """Calls Ollama /api/embed with a batch of texts."""
    payload = {
        "model": model,
        "input": texts,
        "truncate": True,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("embeddings", [])


def export_npz(conn: sqlite3.Connection, npz_path: Path):
    """Exports SQLite embeddings to a high-speed compressed NPZ archive."""
    print(f"\n[Export NPZ] Reading articles from database...")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, url, region, date, topic, title, word_count, embedding 
        FROM articles 
        ORDER BY date ASC, region ASC
    """)
    rows = cursor.fetchall()
    if not rows:
        print("No articles to export.")
        return

    filenames = []
    urls = []
    regions = []
    dates = []
    topics = []
    titles = []
    word_counts = []
    embeddings_list = []

    for r in rows:
        filenames.append(r[0])
        urls.append(r[1])
        regions.append(r[2])
        dates.append(r[3])
        topics.append(r[4])
        titles.append(r[5])
        word_counts.append(r[6])
        vec = np.frombuffer(r[7], dtype=np.float32)
        embeddings_list.append(vec)

    embeddings_matrix = np.vstack(embeddings_list)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        embeddings=embeddings_matrix,
        filenames=np.array(filenames),
        urls=np.array(urls),
        regions=np.array(regions),
        dates=np.array(dates),
        topics=np.array(topics),
        titles=np.array(titles),
        word_counts=np.array(word_counts, dtype=np.int32),
    )
    print(f"[Export NPZ] Saved {len(rows)} articles to {npz_path} (Matrix shape: {embeddings_matrix.shape})")


def export_csv(conn: sqlite3.Connection, csv_path: Path):
    """Exports articles with JSON-string embeddings to CSV."""
    print(f"\n[Export CSV] Reading articles from database...")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, url, region, date, topic, title, word_count, embedding 
        FROM articles 
        ORDER BY date ASC, region ASC
    """)
    rows = cursor.fetchall()
    if not rows:
        print("No articles to export.")
        return

    records = []
    for r in rows:
        vec = np.frombuffer(r[7], dtype=np.float32).tolist()
        records.append({
            "filename": r[0],
            "url": r[1],
            "region": r[2],
            "date": r[3],
            "topic": r[4],
            "title": r[5],
            "word_count": r[6],
            "bge_embedding": json.dumps(vec),
        })

    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_path), index=False)
    print(f"[Export CSV] Saved {len(df)} articles to {csv_path}")


def export_date_region_aggregated(conn: sqlite3.Connection, agg_path: Path):
    """Averages embeddings across articles grouped by (Date, Region)."""
    print(f"\n[Export Aggregated] Aggregating embeddings by Date and Region...")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, region, word_count, embedding 
        FROM articles 
        WHERE date != '' AND region != ''
        ORDER BY date ASC, region ASC
    """)
    rows = cursor.fetchall()
    if not rows:
        print("No articles to aggregate.")
        return

    from collections import defaultdict
    grouped = defaultdict(list)
    word_stats = defaultdict(list)

    for r in rows:
        key = (r[0], r[1])
        vec = np.frombuffer(r[3], dtype=np.float32)
        grouped[key].append(vec)
        word_stats[key].append(r[2])

    records = []
    for (dt, reg), vecs in grouped.items():
        avg_vec = np.mean(vecs, axis=0).tolist()
        records.append({
            "Date": dt,
            "Region": reg,
            "Articles": len(vecs),
            "Avg_Word_Count": float(np.mean(word_stats[(dt, reg)])),
            "bg gme embedding": json.dumps(avg_vec),
        })

    df_agg = pd.DataFrame(records)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(str(agg_path), index=False)
    print(f"[Export Aggregated] Saved {len(df_agg)} (Date, Region) rows to {agg_path}")


def process_zip(
    zip_path: Path,
    db_path: Path,
    npz_path: Path,
    csv_path: Optional[Path] = None,
    agg_path: Optional[Path] = None,
    batch_size: int = 8,
    limit: int = 0,
    max_words: int = 512,
    model: str = MODEL_NAME,
):
    """Main extraction and embedding loop."""
    if not zip_path.exists():
        print(f"Error: Zip file {zip_path} not found.")
        sys.exit(1)

    conn = init_db(db_path)
    completed = get_completed_filenames(conn)
    print("=" * 70)
    print("BGE-M3 News Embedding Pipeline")
    print("=" * 70)
    print(f"Zip source    : {zip_path}")
    print(f"SQLite target : {db_path}")
    print(f"NPZ target    : {npz_path}")
    print(f"Model         : {model} (1024-dimensional dense vectors)")
    print(f"Context cut   : {'Full article' if max_words == 0 else f'First {max_words} words'}")
    print(f"Batch size    : {batch_size}")
    print(f"Already done  : {len(completed):,} articles")
    print("=" * 70)

    # Open zip and list candidates
    with zipfile.ZipFile(str(zip_path), "r") as z:
        all_names = [
            n for n in z.namelist()
            if n.endswith(".txt") and n not in IGNORE_FILES and not n.startswith("__MACOSX")
        ]
        
        pending = [n for n in all_names if n not in completed]
        print(f"Total candidate files: {len(all_names):,}")
        print(f"Pending to embed     : {len(pending):,}")

        if limit > 0:
            pending = pending[:limit]
            print(f"Limiting execution to: {limit:,} articles")

        if not pending:
            print("\nAll articles have already been embedded!")
            export_npz(conn, npz_path)
            if csv_path:
                export_csv(conn, csv_path)
            if agg_path:
                export_date_region_aggregated(conn, agg_path)
            return

        total_pending = len(pending)
        start_time = time.time()
        processed_count = 0

        for i in range(0, total_pending, batch_size):
            batch_names = pending[i : i + batch_size]
            batch_parsed = []

            for name in batch_names:
                try:
                    raw = z.read(name).decode("utf-8", errors="replace")
                    parsed = parse_article(name, raw)
                    if parsed:
                        batch_parsed.append(parsed)
                    else:
                        print(f"Skipping short/non-news file: {name}")
                except Exception as e:
                    print(f"Error reading {name}: {e}")

            if not batch_parsed:
                continue

            # Extract ONLY news bodies (strictly line 6 onwards)
            # Never include metadata lines 1-5
            if max_words > 0:
                texts_to_embed = [
                    " ".join(p["news_body"].split()[:max_words])
                    for p in batch_parsed
                ]
            else:
                texts_to_embed = [p["news_body"] for p in batch_parsed]

            # Generate embeddings
            try:
                embeddings = fetch_embeddings_batch(texts_to_embed, model=model)
                if len(embeddings) != len(batch_parsed):
                    print(f"Warning: Expected {len(batch_parsed)} embeddings, got {len(embeddings)}. Retrying singly...")
                    embeddings = []
                    for t in texts_to_embed:
                        emb = fetch_embeddings_batch([t], model=model)
                        embeddings.append(emb[0])
            except Exception as e:
                print(f"Batch embedding error: {e}. Retrying sequentially...")
                embeddings = []
                for t in texts_to_embed:
                    try:
                        emb = fetch_embeddings_batch([t], model=model)
                        embeddings.append(emb[0])
                    except Exception as inner_e:
                        print(f"Failed to embed text: {inner_e}")
                        embeddings.append(None)

            # Insert into database
            cursor = conn.cursor()
            inserted_in_batch = 0
            for parsed, emb in zip(batch_parsed, embeddings):
                if emb is None:
                    continue
                emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                cursor.execute("""
                    INSERT OR REPLACE INTO articles 
                    (filename, url, region, date, topic, title, word_count, char_count, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    parsed["filename"],
                    parsed["url"],
                    parsed["region"],
                    parsed["date"],
                    parsed["topic"],
                    parsed["title"],
                    parsed["word_count"],
                    parsed["char_count"],
                    emb_bytes
                ))
                inserted_in_batch += 1

            conn.commit()
            processed_count += inserted_in_batch

            # Progress Logging
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = (total_pending - processed_count) / rate if rate > 0 else 0
            curr_pct = (processed_count / total_pending) * 100
            print(
                f"[{processed_count:5d}/{total_pending:5d} ({curr_pct:4.1f}%)] "
                f"Rate: {rate:4.2f} arts/s | "
                f"Elapsed: {elapsed/60:4.1f}m | "
                f"ETA: {remaining/60:4.1f}m"
            )

    export_npz(conn, npz_path)
    if csv_path:
        export_csv(conn, csv_path)
    if agg_path:
        export_date_region_aggregated(conn, agg_path)

    conn.close()
    print("\nAll embedding processes completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Embed news articles from archive.zip using BGE-M3.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH, help="Path to archive.zip")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to output SQLite database")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ_PATH, help="Path to output NPZ file")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to output article CSV")
    parser.add_argument("--agg", type=Path, default=DEFAULT_AGG_PATH, help="Path to output aggregated CSV")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding requests")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of articles to embed (0 for all)")
    parser.add_argument("--max-words", type=int, default=512, help="Max words per article (0 for full article)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name in Ollama")
    parser.add_argument("--export-only", action="store_true", help="Only export existing DB to NPZ/CSV")

    args = parser.parse_args()

    if args.export_only:
        conn = init_db(args.db)
        export_npz(conn, args.npz)
        if args.csv:
            export_csv(conn, args.csv)
        if args.agg:
            export_date_region_aggregated(conn, args.agg)
        conn.close()
        return

    process_zip(
        zip_path=args.zip,
        db_path=args.db,
        npz_path=args.npz,
        csv_path=args.csv,
        agg_path=args.agg,
        batch_size=args.batch_size,
        limit=args.limit,
        max_words=args.max_words,
        model=args.model,
    )


if __name__ == "__main__":
    main()
