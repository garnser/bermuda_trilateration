# database.py
import sqlite3
import json
import math
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

def init_calibration_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_calibration (
            sensor_mac TEXT PRIMARY KEY,
            tx_power REAL NOT NULL,
            samples_count INTEGER NOT NULL DEFAULT 0
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

def update_sensor_tx_power(sensor_mac, tx_power_sample):
    """
    Insert/Update the sensor's tx_power in the sensor_calibration table
    by computing a running average.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tx_power, samples_count
        FROM sensor_calibration
        WHERE sensor_mac = ?
    """, (sensor_mac,))
    row = cursor.fetchone()

    if row is None:
        # No entry yet, insert fresh
        cursor.execute("""
            INSERT INTO sensor_calibration (sensor_mac, tx_power, samples_count)
            VALUES (?, ?, ?)
        """, (sensor_mac, tx_power_sample, 1))
    else:
        old_tx, old_count = row
        new_count = old_count + 1
        new_tx = (old_tx * old_count + tx_power_sample) / new_count
        cursor.execute("""
            UPDATE sensor_calibration
            SET tx_power = ?,
                samples_count = ?
            WHERE sensor_mac = ?
        """, (new_tx, new_count, sensor_mac))

    conn.commit()
    conn.close()

def load_tx_powers_into_sensor_data(sensor_data_dict):
    """
    Load each sensor's calibrated tx_power from sensor_calibration table
    and update sensor_data_dict in-place.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT sensor_mac, tx_power FROM sensor_calibration")
    rows = cursor.fetchall()
    conn.close()

    for mac, tx in rows:
        # If the sensor is known in config.py, override the default tx_power
        if mac in sensor_data_dict:
            sensor_data_dict[mac]["tx_power"] = tx
        # Optionally log or print something
        # print(f"Loaded tx_power={tx:.2f} for sensor={mac}")
