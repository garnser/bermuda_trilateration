# config.py
import numpy as np

# MQTT configuration
MQTT_BROKER = "192.168.68.104"
MQTT_PORT = 1883
MQTT_USERNAME = "ble"
MQTT_PASSWORD = "ble"
MQTT_TOPIC_TEMPLATE = "bermuda/{mac_address}/scanner/+/rssi"

# Model file and RSSI thresholds
MODEL_FILE = "ml_model.pkl"
STRONG_RSSI_THRESHOLD = -70
WEAK_RSSI_THRESHOLD = -90
TX_POWER = -80

# Global state
global_state = {
    "training_mode": False,
    "persistent_id": None,
    "actual_position": None,
}

# Logging configuration
LOGLEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR

# Sensor data
sensor_data = {
    "1c:69:20:cc:f2:c4": {"name": "Kitchen Sensor", "position": (0, 0, 1)},
    "2c:bc:bb:0d:09:74": {"name": "Washroom Sensor", "position": (9, 2.7, 1)},
    "f8:b3:b7:2a:d8:60": {"name": "Living Room Sensor", "position": (8.2, 9.9, 1)},
    "f8:b3:b7:2a:d5:bc": {"name": "Allrum Sensor", "position": (3.8, 5.8, 3.2)},
    "2c:bc:bb:0e:20:88": {"name": "Bedroom Sensor", "position": (8, 0, 4)}
}

# Floor boundaries and room data
floor_boundaries = {
    0: np.array([[10.1, 0.0], [0.0, 0.0], [0.0, 6.34], [3.7, 6.34], [3.7, 10.1], [10.1, 10.1], [10.1, 0.0]]),
    1: np.array([[10.1, 0.0], [0.0, 0.0], [0.0, 6.34], [10.1, 6.34], [10.1, 0.0]]),
}

# Ceiling data
CEILING_HEIGHT = 2.6

room_data = [
    {"name": "Kitchen", "floor": 0, "corners": [(0, 0), (3.2, 0), (3.2, 2.7), (0, 2.7)]},
    {"name": "Dining Area", "floor": 0, "corners": [(0, 2.7), (4.2, 2.7), (5.7, 4), (5.7, 6.34), (0, 6.34)]},
    {"name": "Living Room", "floor": 0, "corners": [(3.7, 6.34), (10.1, 6.34), (10.1, 10.1), (3.7, 10.1)]},
    {"name": "Hallway", "floor": 0, "corners": [(3.2, 0), (7.0, 0), (7.0, 2.7), (8.5, 2.7), (8.5, 4), (5.7, 4), (4.2, 2.7), (3.2, 2.7)]},
    {"name": "Ara room", "floor": 0, "corners": [(7.0, 0), (10.1, 0), (10.1, 2.7), (7.0, 2.7)]},
    {"name": "Toilet", "floor": 0, "corners": [(5.7, 4), (7.6, 4), (7.6, 6.34), (5.7, 6.34)]},
    {"name": "Washroom", "floor": 0, "corners": [(7.6, 4), (8.5, 4), (8.5, 2.7), (10.1, 2.7), (10.1, 6.34), (7.6, 6.34)]},
    {"name": "Melissa room", "floor": 1, "corners": [(0, 0), (0, 2.7), (3.65, 2.7), (3.65, 0)]},
    {"name": "Boys room", "floor": 1, "corners": [(0, 2.7), (0, 6.34), (3.65, 6.34), (3.65, 2.7)]},
    {"name": "TV room", "floor": 1, "corners": [(3.65, 0), (3.65, 6.34), (7, 6.34), (7, 5), (8.3, 5), (8.3, 3.3), (7, 3.3), (3.65, 0)]},
    {"name": "Bedroom", "floor": 1, "corners": [(7, 0), (7, 3.3), (10.1, 3.3), (10.1, 0)]},
    {"name": "Wardrobe", "floor": 1, "corners": [(7, 5), (7, 6.34), (8.3, 6.34), (8.3, 5)]},
    {"name": "Bathroom", "floor": 1, "corners": [(8.3, 3.3), (8.3, 6.34), (10.1, 6.34), (10.1, 3.3)]},
]

# Global RSSI data storage
rssi_data = {}
