# mqtt_client.py
import json
import time
import threading
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, global_state, sensor_data, MQTT_TOPIC_TEMPLATE, rssi_data, STRONG_RSSI_THRESHOLD
from positioning import get_live_position
from database import store_training_data
from logger import logger

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
            rssi_data[persistent_id][sensor_mac] = rssi_value
            # If enough readings exist, update the live position.
            if len(rssi_data[persistent_id]) >= 3:
                _ = get_live_position()
                # In training mode, store the data.
                if global_state.get("training_mode") and global_state.get("actual_position"):
                    inserted = store_training_data(persistent_id, mac_address, global_state["actual_position"], rssi_data)
                    logger.info(f"[INFO] Stored {inserted} training samples.")
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
