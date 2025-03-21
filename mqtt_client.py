# mqtt_client.py
import json
import time
import threading
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, global_state, sensor_data, MQTT_TOPIC_TEMPLATE, rssi_data, STRONG_RSSI_THRESHOLD, TRAIN_EVERY_N_SAMPLES
from utils import estimate_tx_power
from positioning import get_live_position
from database import store_training_data, update_sensor_tx_power
from logger import logger
import numpy as np

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic_parts = msg.topic.split("/")
        if len(topic_parts) < 5:
            return
        sensor_mac = topic_parts[3].lower()
        rssi_value = payload.get("rssi")
        if rssi_value is None or not isinstance(rssi_value, (int, float)):
            return
        persistent_id = payload.get("persistent_id")
        mac_address = payload.get("device")
        if persistent_id is None:
            persistent_id = mac_address
        if sensor_mac in sensor_data:
            if persistent_id not in rssi_data:
                rssi_data[persistent_id] = {}
            # Instead of storing just one value, keep a rolling list (up to 5).
            if sensor_mac not in rssi_data[persistent_id]:
                rssi_data[persistent_id][sensor_mac] = []
            rssi_data[persistent_id][sensor_mac].append(rssi_value)

            # Limit to the last 5 samples (discard older data).
            if len(rssi_data[persistent_id][sensor_mac]) > 5:
                rssi_data[persistent_id][sensor_mac].pop(0)
            # If enough readings exist, update the live position.
            if len(rssi_data[persistent_id]) >= 3:
                # Always update the live position
                estimated_pos = get_live_position()

                # If training mode is active, store multiple snapshots
                if global_state.get("training_mode") and global_state.get("actual_position"):
                    now = time.time()
                    # Check the last store time for this persistent_id
                    if persistent_id not in global_state["last_store_time"]:
                        global_state["last_store_time"][persistent_id] = 0

                    # Example: store a new sample only if at least 2 seconds have passed
                    if now - global_state["last_store_time"][persistent_id] > 2:
                        inserted = store_training_data(
                            persistent_id,
                            mac_address,
                            global_state["actual_position"],
                            rssi_data
                        )
                        global_state["last_store_time"][persistent_id] = now
                        logger.info(f"[INFO] Stored {inserted} training samples (1:many).")

                        actual_pos = np.array(global_state["actual_position"])
                        if sensor_mac in sensor_data:
                            sensor_pos = np.array(sensor_data[sensor_mac]["position"])
                            distance = np.linalg.norm(sensor_pos - actual_pos)
                            tx_power_sample = estimate_tx_power(rssi_value, distance, n=2.0)
                            if tx_power_sample is not None:
                                update_sensor_tx_power(sensor_mac, tx_power_sample)
                                logger.debug(f"[DEBUG] Updated tx_power for {sensor_mac} with sample {tx_power_sample:.2f}")

                        # After storing a sample, possibly re-train the model.
                        global_state.setdefault("samples_since_train", 0)
                        global_state["samples_since_train"] += inserted
                        if global_state["samples_since_train"] >= TRAIN_EVERY_N_SAMPLES:
                            _retrain_ml_model()

                # In training mode, store the data.
                if global_state.get("training_mode") and global_state.get("actual_position"):
                    inserted = store_training_data(persistent_id, mac_address, global_state["actual_position"], rssi_data)
                    logger.info(f"[INFO] Stored {inserted} training samples.")

                    global_state.setdefault("samples_since_train", 0)
                    global_state["samples_since_train"] += inserted
                    if global_state["samples_since_train"] >= TRAIN_EVERY_N_SAMPLES:
                        _retrain_ml_model()
    except Exception as e:
        logger.error(f"[ERROR] MQTT message error: {e}")

def connect_mqtt():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    return client

def start_mqtt_listener(persistent_id):
    client = connect_mqtt()
    topic = MQTT_TOPIC_TEMPLATE.format(mac_address=persistent_id)
    client.subscribe(topic)
    logger.info(f"[INFO] Subscribed to {topic}")
    client.loop_start()
    while True:
        time.sleep(1)

def _retrain_ml_model():
    """
    Helper to re-train the ML model and reset the sample counter.
    """
    from ml_model import train_ml_model
    logger.info("[INFO] Re-training ML model with new samples...")
    train_ml_model(rssi_data)
    global_state["samples_since_train"] = 0
