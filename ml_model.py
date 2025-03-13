# ml_model.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import threading
import sqlite3
import json
from config import MODEL_FILE, STRONG_RSSI_THRESHOLD, WEAK_RSSI_THRESHOLD, sensor_data
from database import DB_FILE
from utils import rssi_to_distance
from logger import logger

model_lock = threading.Lock()
model = None
sensor_macs = None
label_encoder = None

def load_ml_model():
    global model, sensor_macs, label_encoder
    try:
        with open(MODEL_FILE, "rb") as f:
            model, sensor_macs, label_encoder = pickle.load(f)
        logger.info(f"[INFO] Loaded ML model from {MODEL_FILE}")
    except FileNotFoundError:
        logger.warning("[WARNING] No saved ML model found. Model will be trained when needed.")

def train_ml_model(rssi_data):
    """Train a RandomForest model using training data from the database."""
    global model, label_encoder, sensor_macs
    with model_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT persistent_id, mac_address, rssi_json, x, y, z, score FROM training_data")
        data = cursor.fetchall()
        conn.close()
        if len(data) < 3:
            logger.warning("[WARNING] Not enough training data. Skipping training.")
            label_encoder = LabelEncoder()
            label_encoder.fit(["generic"])
            return None

        sensor_macs = sorted(sensor_data.keys())
        devices = [row[0] for row in data]
        features = []
        labels = []
        sample_weights = []
        for persistent_id, mac_address, rssi_json, x, y, z, score in data:
            rssi_dict = json.loads(rssi_json)
            row_features = []
            sensor_weights = {}
            for mac in sensor_macs:
                sensor_rssi = rssi_dict.get(mac, -100)
                sensor_pos = sensor_data.get(mac, {}).get("position", (0, 0, 0))
                if sensor_rssi >= STRONG_RSSI_THRESHOLD:
                    weight = 1.5
                elif sensor_rssi < WEAK_RSSI_THRESHOLD:
                    weight = 0.3
                else:
                    weight = 1.0
                sensor_weights[mac] = weight
                row_features.extend([sensor_rssi * weight, sensor_pos[0], sensor_pos[1], sensor_pos[2]])
            for i in range(len(sensor_macs)):
                for j in range(i + 1, len(sensor_macs)):
                    rssi_i = rssi_dict.get(sensor_macs[i], -100)
                    rssi_j = rssi_dict.get(sensor_macs[j], -100)
                    weight_i = sensor_weights.get(sensor_macs[i], 1.0)
                    weight_j = sensor_weights.get(sensor_macs[j], 1.0)
                    row_features.append(rssi_i - rssi_j)
            features.append(row_features)
            labels.append([x, y, z])
            sample_weights.append(max(score if score is not None else 1.0, 0.1))

        features = np.array(features)
        labels = np.array(labels)

        # DEBUG: Log details about the training data.
        logger.debug("[DEBUG] Training data samples: %s", features.shape[0])
        logger.debug("[DEBUG] Features shape: %s", features.shape)
        logger.debug("[DEBUG] Labels shape: %s", labels.shape)
        logger.debug("[DEBUG] Features min: %s max: %s", np.nanmin(features), np.nanmax(features))
        logger.debug("[DEBUG] Labels min: %s max: %s", np.nanmin(labels), np.nanmax(labels))
        logger.debug("[DEBUG] Sample weights: %s", sample_weights)
        if np.any(np.isnan(features)):
            logger.error("[ERROR] NaN detected in training features!")
        if np.any(np.isnan(labels)):
            logger.error("[ERROR] NaN detected in training labels!")

        # Include a default "generic" label
        all_labels = set(devices)
        all_labels.add("generic")
        all_labels = sorted(all_labels)
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        encoded_devices = label_encoder.transform(devices)
        features = np.column_stack((features, encoded_devices))
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, labels, sample_weight=sample_weights)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, sensor_macs, label_encoder), f)
        logger.info(f"[INFO] ML model trained on {len(features)} samples and saved to {MODEL_FILE}")
        return model, sensor_macs, label_encoder
