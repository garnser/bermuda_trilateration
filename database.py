# database.py
import sqlite3

DB_FILE = "training_data.db"

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persistent_id TEXT NOT NULL,
            mac_address TEXT,
            sensor_mac TEXT,
            rssi FLOAT,
            x FLOAT,
            y FLOAT,
            z FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def store_training_data(persistent_id, mac_address, estimated_position, rssi_data):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    inserted_count = 0
    for sensor_mac, rssi in rssi_data.get(persistent_id, {}).items():
        if rssi is None or not isinstance(rssi, (int, float)):
            continue
        cursor.execute("""
            INSERT INTO training_data (persistent_id, mac_address, sensor_mac, rssi, x, y, z)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (persistent_id, mac_address, sensor_mac, float(rssi),
              estimated_position[0], estimated_position[1], estimated_position[2]))
        inserted_count += 1
    conn.commit()
    conn.close()
    return inserted_count
