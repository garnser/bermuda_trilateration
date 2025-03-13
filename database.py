# database.py
import sqlite3
import json
from config import global_state

DB_FILE = "training_data.db"

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persistent_id TEXT NOT NULL,
            mac_address TEXT,
            rssi_json TEXT,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            score FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def store_training_data(persistent_id, mac_address, estimated_position, rssi_data):
    score = global_state.get("training_score", None)
    aggregated_rssi = rssi_data.get(persistent_id, {})
    rssi_json = json.dumps(aggregated_rssi)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO training_data (persistent_id, mac_address, rssi_json, x, y, z, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (persistent_id, mac_address, rssi_json,
              estimated_position[0], estimated_position[1], estimated_position[2], score))
    conn.commit()
    conn.close()
    return 1
