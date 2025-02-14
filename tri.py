import yaml
import sqlite3
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
import argparse
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patches as mpatches
import random
import hashlib
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import threading
import time

MQTT_BROKER = "192.168.68.104"
MQTT_PORT = 1883
MQTT_USERNAME = "ble"
MQTT_PASSWORD = "ble"
MQTT_TOPIC_TEMPLATE = "bermuda/{mac_address}/scanner/+/rssi"

global_state = {
    "training_mode": False,
    "persistent_id": None,
    "actual_position": None,
}

MODEL_FILE = "ml_model.pkl"

model = None
label_encoder = None
sensor_macs = []
new_readings_count = 0

# Load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return {k.lower(): v for k, v in data.items()}  # Normalize MAC addresses to lowercase

def init_database():
    """Initialize SQLite database and ensure all necessary columns exist."""
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    # **Create training_data table (if not exists)**
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        persistent_id TEXT NOT NULL,
        mac_address TEXT,
        rssi_json TEXT,  -- Stores multiple RSSI readings as JSON
        x FLOAT DEFAULT NULL,
        y FLOAT DEFAULT NULL,
        z FLOAT DEFAULT NULL,
        weight INTEGER DEFAULT 3,  -- Default weight for training data
        position_available INTEGER DEFAULT 1,  -- 1 if (x, y, z) exists, 0 if RSSI-only
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # **Create sensor_positions table (if not exists)**
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_positions (
        mac_address TEXT PRIMARY KEY,
        x FLOAT NOT NULL,
        y FLOAT NOT NULL,
        z FLOAT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

def load_config(file_path="config.yaml"):
    """Loads sensor data, room data, and floor boundaries from YAML config."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Convert sensor positions from lists to tuples
    config["sensors"] = {mac.lower(): {"name": sensor["name"], "position": tuple(sensor["position"])}
                         for mac, sensor in config["sensors"].items()}

#    print("Loaded sensor_data:", json.dumps(config["sensors"], indent=4))

    # Convert floor boundaries to NumPy arrays
    config["floor_boundaries"] = {int(k): np.array(v) for k, v in config["floor_boundaries"].items()}

    return config

# Load configuration at startup
#config = load_config()

#sensor_data = config["sensors"]
#room_data = config["rooms"]
#floor_boundaries = config["floor_boundaries"]

def load_ml_model():
    global model, sensor_macs, label_encoder
    sensor_macs = []  # Default empty list

    try:
        with open(MODEL_FILE, "rb") as f:
            model, loaded_sensor_macs, label_encoder = pickle.load(f)
            sensor_macs = loaded_sensor_macs if loaded_sensor_macs else list(sensor_data.keys())

#        print(f"✅ Loaded ML model from {MODEL_FILE}")
    except FileNotFoundError:
        print("⚠️ No saved ML model found. Loading sensor MACs from sensor_data...")
        sensor_macs = list(sensor_data.keys())
        model = None
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        model = None

def fetch_sensor_macs_from_db():
    """Fetch distinct sensor MAC addresses from the database."""
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT mac_address FROM training_data")
    macs = [row[0] for row in cursor.fetchall()]

    conn.close()

#    print(f"✅ Loaded {len(macs)} sensor MACs from database.")
    return macs

def validate_rssi_data(persistent_id, rssi_json):
    """Ensure we have enough valid RSSI readings for training or prediction."""
    if not rssi_json:
        print(f"⚠️ Skipping {persistent_id}: No RSSI data found in the database!")
        return False

#    print(f"📊 Validating RSSI data for {persistent_id}: {rssi_json}")

    rssi_dict = json.loads(rssi_json)
    valid_rssi_values = [rssi for rssi in rssi_dict.values() if rssi > -100]  # Ignore -100 values

    if len(valid_rssi_values) < 3:
#        print(f"⚠️ Skipping {persistent_id}: Too few RSSI readings available ({len(valid_rssi_values)} found, need at least 3).")
        return False

#    print(f"📊 Parsed RSSI dictionary: {rssi_dict}")

    valid_rssi_values = [rssi for rssi in rssi_dict.values() if isinstance(rssi, (int, float)) and rssi > -100]

#    print(f"✅ {persistent_id} has {len(valid_rssi_values)} valid RSSI readings.")

    if all(rssi == -100 for rssi in rssi_dict.values()):
        print(f"⚠️ Skipping {persistent_id}: All RSSI values are invalid (-100 readings).")
        return False

    return True

import numpy as np

def train_ml_model():
    global model, label_encoder, sensor_macs

    print("🔄 Starting ML model training...")
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    # **Always reload sensor MACs from the database**
    sensor_macs = list(sensor_data.keys())

    if not sensor_macs:
        print("❌ No sensor MACs found! Training aborted.")
        return

    cursor.execute("""
        SELECT persistent_id, mac_address, rssi_json, x, y, z, weight
        FROM training_data
        WHERE position_available = 1 OR weight = 2
    """)
    data_batch = cursor.fetchall()

    if not data_batch:
        print("⚠️ No valid training data found. Skipping ML training.")
        conn.close()
        return

    print(f"✅ Found {len(data_batch)} training samples.")

    features, labels, weights = [], [], []

    for persistent_id, mac_address, rssi_json_str, x, y, z, weight in data_batch:
        if not validate_rssi_data(persistent_id, rssi_json_str):
            continue

        try:
            rssi_dict = json.loads(rssi_json_str)
        except Exception:
            print(f"❌ JSON decoding error for {persistent_id}")
            continue

        # **Ensure RSSI vector uses the same sensors from the database**
        row_rssi = np.array([rssi_dict.get(mac, np.nan) for mac in sensor_macs])  # Use NaN instead of -100

        if np.all(np.logical_or(np.isnan(row_rssi), row_rssi == -100)):
            continue

        # **Ensure position values are valid**
        try:
            pos = [float(x), float(y), float(z)]
        except Exception as e:
            print(f"❌ Skipping {persistent_id}: Could not convert position values to float: {x}, {y}, {z}")
            continue

        if any(not np.isfinite(v) for v in pos):
            print(f"⚠️ Skipping {persistent_id}: Invalid position values (x={x}, y={y}, z={z})")
            continue

        features.append(row_rssi)
        labels.append(pos)
        weights.append(weight)

    conn.close()

    if not features:
        print("❌ No valid features found! Training aborted.")
        return

    features = np.array(features)
    labels = np.array(labels)
    weights = np.array(weights)

    print(f"✅ Training ML model with {len(features)} samples.")

    try:
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

        print("🔍 Checking labels for NaN or infinite values...")
        if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
            print(f"❌ ERROR: NaN or Inf detected in labels! Aborting training.")
            return

        model.fit(features, labels, sample_weight=weights)
        print(f"✅ ML model trained on {len(features)} samples.")

        # **Save the updated model along with sensor_macs**
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, sensor_macs, label_encoder), f)

    except Exception as e:
        print(f"❌ Error during ML training: {e}")

def generate_room_color(room_name):
    """Generate a stable color for a room based on its name."""
    hash_val = int(hashlib.md5(room_name.encode()).hexdigest(), 16)  # Convert room name to a hash
    r = (hash_val % 256) / 255  # Extract Red component
    g = ((hash_val >> 8) % 256) / 255  # Extract Green component
    b = ((hash_val >> 16) % 256) / 255  # Extract Blue component
    return (r, g, b, 1)  # Use alpha=0.5 for transparency

# Trilateration function
def trilaterate(sensors, distances):
    """Compute device position based on sensor positions and estimated distances."""

    def residuals(pos, sensors, distances, weights):
        """Calculate the error between estimated and actual distances."""
        return weights * (np.linalg.norm(pos - np.array(sensors), axis=1) - distances)

    print(f"📡 Trilateration Sensors (Positions): {sensors}")
    print(f"📏 Trilateration Distances (Meters): {distances}")

    if len(sensors) < 3:
        print("⚠️ Not enough sensors for trilateration.")
        return None

    sensors = np.array(sensors)
    distances = np.array(distances)

    # Fix: Ensure distances are within a reasonable range
    distances = np.clip(distances, -2, 13)  # Min 0.5m, Max 15m

    # Fix: Ensure Z-axis influence is weighted properly
    weights = 1 / (np.square(distances) + 0.1)  # Avoid division by zero

    initial_guess = np.mean(sensors, axis=0)

    try:
        result = least_squares(residuals, initial_guess, args=(sensors, distances, weights))
        estimated_position = result.x

        # Fix: Ensure estimated Z value is within bounds
        estimated_position[2] = max(0, min(estimated_position[2], 6))  # Z should be between 0-3 meters

        print(f"✅ Trilateration Result: {estimated_position}")
        return estimated_position
    except Exception as e:
        print(f"❌ Trilateration Error: {e}")
        return None

def get_live_position():
    global rssi_data, model, sensor_macs

    if not rssi_data:
        print("⚠️ No RSSI data available.")
        return None

    persistent_id = global_state.get("persistent_id")
    if not persistent_id or persistent_id not in rssi_data:
        print(f"⚠️ Persistent ID {persistent_id} not recognized in RSSI data.")
        return None

#    print(f"📡 Fetching RSSI data for {persistent_id}: {rssi_data[persistent_id]}")

    # **Always use the same sensor MACs from the trained model**
    if model is None or not sensor_macs:
        print("⚠️ No trained ML model available. Retraining now...")
        train_ml_model()

    if model is None or not sensor_macs:
        print("❌ No valid model available. Skipping prediction.")
        return None

    # **Build RSSI vector with consistent sensor order**
    rssi_values = np.array([rssi_data[persistent_id].get(mac, -100) for mac in sensor_macs])

    if len(rssi_values) != len(sensor_macs):
        print(f"❌ Feature shape mismatch: Expected {len(sensor_macs)}, Got {len(rssi_values)}")
        return None

#    print(f"🧐 RSSI Vector for ML Model: {rssi_values}")

    try:
        predicted_position = model.predict(rssi_values.reshape(1, -1))[0]
        print(f"🔍 ML Predicted Position for {persistent_id} → {predicted_position}")
        return predicted_position
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def store_training_data(persistent_id, mac_address, estimated_position=None, rssi_snapshot=None):
    """Store RSSI-based training data in SQLite for future ML training."""
    global new_readings_count

    if rssi_snapshot is None:
        rssi_snapshot = rssi_data.get(persistent_id, {})

    valid_rssi_readings = {k: v for k, v in rssi_snapshot.items() if v != -100}
    if len(valid_rssi_readings) < 3:
#        print(f"⚠️ Not enough valid RSSI readings for {persistent_id}. Skipping storage.")
        return

    # Extract last octet of beacon MAC and generate its expected sensor MAC
    octets = mac_address.split(":")
    last_octet = int(octets[-1], 16)
    expected_sensor_mac = ":".join(octets[:-1] + [f"{(last_octet - 2) % 256:02x}"])

    if global_state["training_mode"] and persistent_id == global_state["persistent_id"]:
        estimated_position = global_state["actual_position"]
        weight = 3  # training sample
    elif expected_sensor_mac in sensor_data:
        estimated_position = sensor_data[expected_sensor_mac]["position"]
        weight = 2  # sensor-based reading
    else:
        return

    new_readings_count += 1
    if new_readings_count >= 100:
        print("🔄 Retraining model after 1000 new readings.")
        train_ml_model()
        new_readings_count = 0

    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    rssi_json = json.dumps(valid_rssi_readings)
    try:
        x, y, z = map(float, estimated_position)
    except Exception as e:
        print(f"❌ Error converting position to float for {persistent_id}: {estimated_position}")
        return

    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
        print(f"❌ Invalid position detected: x={x}, y={y}, z={z}. Skipping.")
        conn.close()
        return

    if global_state["training_mode"] or expected_sensor_mac in sensor_data:
        try:
            cursor.execute("""
            INSERT INTO training_data (persistent_id, mac_address, rssi_json, x, y, z, weight, position_available)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (persistent_id, mac_address, rssi_json, x, y, z, 3, 1))

            conn.commit()
            print(f"✅ Training data stored for {persistent_id} (x={x}, y={y}, z={z})")

        except sqlite3.Error as e:
            print(f"❌ SQLite Error: {e}")
        finally:
            conn.close()

def store_sensor_positions():
    """Store the known sensor positions in the database for reference."""
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_positions (
        mac_address TEXT PRIMARY KEY,
        x FLOAT NOT NULL,
        y FLOAT NOT NULL,
        z FLOAT NOT NULL
    );
    """)

    for mac, data in sensor_data.items():
        x, y, z = data["position"]
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO sensor_positions (mac_address, x, y, z)
            VALUES (?, ?, ?, ?)
            """, (mac, x, y, z))
        except sqlite3.Error as e:
            print(f"❌ SQLite Error storing sensor positions: {e}")

    conn.commit()
    conn.close()

def smooth_rssi(persistent_id, sensor_mac, new_rssi, alpha=0.2):
    """Applies exponential smoothing to reduce RSSI fluctuations."""
    if persistent_id not in rssi_data:
        rssi_data[persistent_id] = {}

    if sensor_mac in rssi_data[persistent_id]:
        previous_rssi = rssi_data[persistent_id][sensor_mac]
        smoothed_rssi = alpha * new_rssi + (1 - alpha) * previous_rssi  # Exponential smoothing formula
    else:
        smoothed_rssi = new_rssi  # No previous value, use new RSSI

    return smoothed_rssi

def on_message(client, userdata, msg):
    global new_readings_count
    try:
        payload_str = msg.payload.decode(errors='ignore')
        payload = json.loads(payload_str)

        persistent_id = payload.get("persistent_id", payload.get("device", None))
        if not persistent_id:
            print(f"⚠️ Missing persistent_id! Skipping message.")
            return

        beacon_mac = payload.get("device", "").lower()
        scanner_mac = payload.get("scanner", "").split(" ")[-1].strip("()").lower()
        new_rssi = payload.get("rssi")

        if new_rssi is None or not isinstance(new_rssi, (int, float)):
            print(f"⚠️ Invalid RSSI value: {new_rssi}. Skipping.")
            return

#        print(f"📩 New RSSI: PersistentID={persistent_id}, Scanner={scanner_mac}, RSSI={new_rssi}")  # Debugging

        smoothed_rssi = smooth_rssi(persistent_id, scanner_mac, new_rssi)
        rssi_data.setdefault(persistent_id, {})[scanner_mac] = smoothed_rssi

#        print(f"📊 Updated RSSI Data: {rssi_data[persistent_id]}")  # Debugging

        # **Process Data**
        estimated_position = get_live_position()
        store_training_data(persistent_id, beacon_mac, estimated_position, rssi_snapshot=rssi_data.get(persistent_id))

    except Exception as e:
        print(f"❌ MQTT Error: {e}")

def downsample_database():
    """Keeps the database size manageable by removing older data."""
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    # Determine number of rows
    cursor.execute("SELECT COUNT(*) FROM training_data")
    total_rows = cursor.fetchone()[0]

    # Keep only the latest 10,000 records
    max_rows = 10000
    if total_rows > max_rows:
        delete_rows = total_rows - max_rows
        print(f"🗑️ Downsampling: Removing {delete_rows} old records...")
        cursor.execute("""
            DELETE FROM training_data WHERE id IN (
                SELECT id FROM training_data ORDER BY timestamp ASC LIMIT ?
            )
        """, (delete_rows,))
        conn.commit()

    conn.close()

# **MQTT Authentication & Connection**
def connect_mqtt():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  # Set authentication
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    return client

# Start MQTT Listener
def start_mqtt_listener(update_interval=2):
    """Start MQTT listener and periodically fetch live position."""
    client = connect_mqtt()
    topic = f"bermuda/+/scanner/+/rssi"
    client.subscribe(topic)
    print(f"Subscribed to {topic}")

    client.loop_start()

    while True:
        time.sleep(1)

floor_boundaries = {
    0: np.array([[10.1, 0.0], [0.0, 0.0], [0.0, 6.34], [3.7, 6.34], [3.7, 10.1], [10.1, 10.1], [10.1, 0.0]]),  # Ground floor
    1: np.array([[10.1, 0.0], [0.0, 0.0], [0.0, 6.34], [10.1, 6.34], [10.1, 0.0]]),
}

# Define room positions and sizes
room_data = [
    {"name": "Kitchen", "floor": 0, "corners": [(0, 0), (3.2, 0), (3.2, 2.7), (0, 2.7)]},
    {"name": "Dining Area", "floor": 0, "corners": [(0, 2.7), (4.2, 2.7), (5.7, 4), (5.7, 6.34), (0, 6.34)]},
    {"name": "Living Room", "floor": 0, "corners": [(3.7, 6.34), (10.1, 6.34), (10.1, 10.1), (3.7, 10.1)]},
    {"name": "Hallway", "floor": 0, "corners": [(3.2,0), (7.0, 0), (7.0, 2.7), (8.5, 2.7), (8.5, 4), (5.7, 4.0), (4.2, 2.7), (3.2, 2.7)]},
    {"name": "Ara room", "floor": 0, "corners": [(7.0, 0), (10.1, 0), (10.1, 2.7), (7.0, 2.7)]},
    {"name": "Toilet", "floor": 0, "corners": [(5.7, 4), (7.6, 4), (7.6, 6.34), (5.7, 6.34)]},
    {"name": "Washroom", "floor": 0, "corners": [(7.6, 4), (8.5, 4), (8.5, 2.7), (10.1, 2.7), (10.1, 6.34), (7.6, 6.34)]},
    {"name": "Melissa room", "floor": 1, "corners": [(0,0), (0, 2.7), (3.65, 2.7), (3.65, 0)]},
    {"name": "Boys room", "floor": 1, "corners": [(0, 2.7), (0, 6.34), (3.65, 6.34), (3.65, 2.7)]},
    {"name": "TV room", "floor": 1, "corners": [(3.65, 0), (3.65, 6.34), (7, 6.34), (7, 5), (8.3, 5), (8.3, 3.3), (7, 3.3), (7, 0)]},
    {"name": "Bedroom", "floor": 1, "corners": [(7, 0), (7, 3.3), (10.1, 3.3), (10.1, 0)]},
    {"name": "Wardrobe", "floor": 1, "corners": [(7, 5), (7, 6.34), (8.3, 6.34), (8.3, 5)]},
    {"name": "Bathroom", "floor": 1, "corners": [(8.3, 3.3), (8.3, 6.34), (10.1, 6.34), (10.1, 3.3)]},
]

rssi_data = {}
sensor_data = {
    "1c:69:20:cc:f2:c4": {"name": "Kitchen Sensor", "position": (0, 0, 1)},
    "2c:bc:bb:0d:09:74": {"name": "Washroom Sensor", "position": (9, 2.7, 1)},
    "f8:b3:b7:2a:d8:60": {"name": "Living Room Sensor", "position": (8.2, 9.9, 1)},
    "f8:b3:b7:2a:d5:bc": {"name": "Allrum Sensor", "position": (3.8, 5.8, 3.2)},
    "2c:bc:bb:0e:20:88": {"name": "Bedroom Sensor", "position": (8, 0, 4)}
}

# Generate unique colors for rooms
room_colors = {room["name"]: generate_room_color(room["name"]) for room in room_data}

def remove_overlapping_ceiling(floor_boundaries):
    """Removes overlapping sections of the ceiling based on the floor above."""
    ceiling_polygons = {}

    for floor in floor_boundaries:
        if floor + 1 in floor_boundaries:  # Check if there's a floor above
            lower_floor_polygon = Polygon(floor_boundaries[floor])
            upper_floor_polygon = Polygon(floor_boundaries[floor + 1])
            non_overlapping_ceiling = lower_floor_polygon.difference(upper_floor_polygon)
            ceiling_polygons[floor] = non_overlapping_ceiling
        else:
            ceiling_polygons[floor] = Polygon(floor_boundaries[floor])  # No floor above, keep full ceiling

    return ceiling_polygons

def find_room(estimated_position):
    """Determine which room the estimated position is in."""
    estimated_point = Point(estimated_position[:2])
    estimated_floor = int(estimated_position[2] // 3)  # Assuming each floor is 3m high

    for room in room_data:
        if room["floor"] == estimated_floor:
            room_polygon = Polygon(room["corners"])
            if room_polygon.contains(estimated_point):
                return room["name"]

    return "Unknown Room"

# Visualization function
def plot_3d_position(sensors, estimated_position):
    plt.ion()  # Enable interactive mode for live updates

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    print("🎥 plot_3d_position() has started... waiting for updates...")

    while True:  # Continuous update loop
        ax.clear()  # Clear previous frame

        # **Get live estimated position from MQTT**
#        print("🟢 Calling get_live_position() to fetch latest estimated position...")
        estimated_position = get_live_position()  # Function to retrieve latest calculated position

        if estimated_position is None:
            print("⚠️ Estimated position is None. Skipping plot update.")
            time.sleep(1)
            continue

        print(f"🔴 Plotting estimated position: {estimated_position}")

        # Get max floor height dynamically
        floor_heights = {room["floor"]: room["floor"] * 3 for room in room_data}
        max_floor = max(floor_heights.keys())
        max_floor_height = floor_heights[max_floor]

        # Compute non-overlapping ceiling sections
        non_overlapping_ceilings = remove_overlapping_ceiling(floor_boundaries)
        custom_legend_handles = []

        for floor, boundary in floor_boundaries.items():
            floor_z = floor_heights.get(floor, 0)

            # Draw floor outline
            ax.plot(boundary[:, 0], boundary[:, 1], np.full_like(boundary[:, 0], floor_z),
                    color='black', linestyle='dashed', label=f'Floor {floor}')

            # Draw walls between floors
            for i in range(len(boundary)):
                ax.plot([boundary[i, 0], boundary[i, 0]],
                        [boundary[i, 1], boundary[i, 1]],
                        [floor_z, floor_z + 3], color='black', linestyle='dotted')

            # Draw non-overlapping ceiling
            if floor in non_overlapping_ceilings and not non_overlapping_ceilings[floor].is_empty:
                x, y = non_overlapping_ceilings[floor].exterior.xy
                ceiling_corners = [(x[i], y[i], (floor + 1) * 3) for i in range(len(x))]
                ceiling_surface = Poly3DCollection([ceiling_corners], color='gray', alpha=0.0, label="Ceiling")
                ax.add_collection3d(ceiling_surface)

        # **Draw rooms as flat surfaces with custom shapes**
        for room in room_data:
            floor_z = floor_heights[room["floor"]]
            room_corners = [(x, y, floor_z) for x, y in room["corners"]]

            # Draw the custom-shaped room
            room_surface = Poly3DCollection([room_corners], color=room_colors[room["name"]], alpha=0.5)
            ax.add_collection3d(room_surface)

            # Add to legend
            custom_legend_handles.append(mpatches.Patch(color=room_colors[room["name"]], label=f'{room["name"]} (Floor {room["floor"]})'))

        # **Draw ceilings**
        for floor in floor_boundaries:
            ceiling_corners = [(x, y, (floor + 1) * 3) for x, y in floor_boundaries[floor]]
            ceiling_surface = Poly3DCollection([ceiling_corners], color='gray', alpha=0.3, label="Ceiling")
            ax.add_collection3d(ceiling_surface)

        # **Plot sensors**
        for mac, data in sensor_data.items():
            pos = data["position"]
            name = data["name"]
            ax.scatter(*pos, c='blue', marker='o', s=80, label="Sensor" if mac == list(sensor_data.keys())[0] else "")
            ax.text(pos[0], pos[1], pos[2], name, color='black', fontsize=8)

        # **Plot estimated position**
        if estimated_position is not None:
            ax.scatter(*estimated_position, c='red', marker='x', s=100, label='Estimated Device Position')
            room_name = find_room(estimated_position)
            ax.text(estimated_position[0] + 0.3, estimated_position[1], estimated_position[2], room_name, color='red', fontsize=10, fontweight='bold')

        # **Restore legend with rooms + sensors + estimated position**
        ax.legend(handles=custom_legend_handles + [
            mpatches.Patch(color='blue', label="Sensor"),
            mpatches.Patch(color='red', label="Estimated Position"),
            mpatches.Patch(color='black', label="House Boundaries"),
            mpatches.Patch(color='gray', alpha=0.3, label="Ceiling")
        ], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.draw()
        plt.pause(2)  # Refresh every 2 seconds

# Main function to process data
def get_device_position(yaml_file, sensor_data, mac_address):
    data = load_yaml(file_path=yaml_file)
    mac_address = mac_address.lower()  # Normalize input MAC address

    if mac_address not in data:
#        print(f"MAC address {mac_address} not found in dataset.")
        return None

    device_data = data[mac_address].get('scanners', {})
    readings = []
    sensor_names = []

    for sensor_mac, sensor_info in device_data.items():
        source_address = sensor_info.get('source', '').lower()  # Normalize source MAC address
        distance = sensor_info.get('rssi_distance')

        if distance is not None and source_address in sensor_data:
            readings.append((sensor_data[source_address]["position"], distance))
            sensor_names.append(sensor_data[source_address]["name"])

    if len(readings) < 3:
#        print(f"Not enough valid sensor readings for MAC {mac_address} to perform trilateration.")
        return None

    sensors, distances = zip(*readings)
    estimated_position = trilaterate(np.array(sensors), np.array(distances))
    plot_3d_position(sensors, estimated_position)
    return estimated_position

# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trilateration with Machine Learning training")
    parser.add_argument("persistent_id", help="Persistent ID of the device to locate")
    parser.add_argument("--train", nargs=3, type=float, help="Provide actual X, Y, Z position for training")
    args = parser.parse_args()

    init_database()  # Ensure database exists
    load_ml_model()  # ✅ Load ML model at startup

    global_state["training_mode"] = args.train is not None
    global_state["persistent_id"] = args.persistent_id

    if global_state["training_mode"]:
        global_state["actual_position"] = tuple(args.train)

    # Start MQTT listener in a separate thread
    mqtt_thread = threading.Thread(target=start_mqtt_listener,)
    mqtt_thread.daemon = True
    mqtt_thread.start()

    # Start 3D visualization with live updates
    plot_3d_position(list(sensor_data.values()), None)
