#!/usr/bin/env python3
"""
Query logger: records every search for autopoiesis analysis.
"""
import json, os, time, sqlite3
from datetime import datetime

DB_PATH = 'logs/query_log.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            query_text TEXT,
            query_len INTEGER,
            num_words INTEGER,
            top_k INTEGER,
            latency_us REAL,
            result_count INTEGER,
            hit_fast_path INTEGER,
            hit_layer TEXT,
            r1_match INTEGER,
            r5_match INTEGER,
            perturbation_type TEXT,
            threshold_1 REAL,
            threshold_2 REAL,
            threshold_3 REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_query(query_text, latency_us, results, top_k, r1_match=0, r5_match=0, 
              perturbation_type='normal', thresholds=None):
    """Call this from engine.search() after every query."""
    os.makedirs('logs', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    words = query_text.split()
    c.execute('''
        INSERT INTO queries 
        (timestamp, query_text, query_len, num_words, top_k, latency_us,
         result_count, hit_fast_path, r1_match, r5_match, perturbation_type,
         threshold_1, threshold_2, threshold_3)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        query_text,
        len(query_text),
        len(words),
        top_k,
        latency_us,
        len(results),
        1 if latency_us < 50 else 0,
        r1_match,
        r5_match,
        perturbation_type,
        thresholds.get('l1', 0.8) if thresholds else 0.8,
        thresholds.get('l2', 0.7) if thresholds else 0.7,
        thresholds.get('l3', 0.6) if thresholds else 0.6,
    ))
    conn.commit()
    conn.close()

def get_stats(hours=24):
    """Return aggregate stats for autopoiesis."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT 
            AVG(latency_us),
            AVG(r1_match),
            AVG(CASE WHEN num_words < 5 THEN r1_match END),
            AVG(CASE WHEN num_words >= 5 THEN r1_match END),
            COUNT(*)
        FROM queries
        WHERE timestamp > datetime('now', '-{} hours')
    '''.format(hours))
    row = c.fetchone()
    conn.close()
    return {
        'avg_latency_us': row[0],
        'avg_r1': row[1],
        'short_query_r1': row[2],
        'long_query_r1': row[3],
        'count': row[4],
    }

if __name__ == '__main__':
    init_db()
    print("Query log DB initialized")
