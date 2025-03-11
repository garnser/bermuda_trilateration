import yaml
import sqlite3
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
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
model_lock = threading.Lock()

model = None
label_encoder = None

# Load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return {k.lower(): v for k, v in data.items()}  # Normalize MAC addresses to lowercase

def init_database():
    """Initialize SQLite database for ML training."""
    conn = sqlite3.connect("training_data.db")
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

def load_ml_model():
    global model, sensor_macs, label_encoder
    try:
        with open(MODEL_FILE, "rb") as f:
            model, sensor_macs, label_encoder = pickle.load(f)
        print(f"✅ Loaded ML model from {MODEL_FILE}")
    except FileNotFoundError:
        print("⚠️ No saved ML model found. Model will be trained when needed.")

def rssi_to_distance(rssi, tx_power=-70, n=2.0):
    """
    Convert RSSI to distance (in meters) using a simple path-loss model.
    tx_power: RSSI at 1 meter (default: -59 dBm)
    n: path-loss exponent (default: 2.0 for free space)
    """
    return 10 ** ((tx_power - rssi) / (10 * n))

def train_ml_model():
    """Train a machine learning model using all available RSSI data from all devices."""
    global model, label_encoder, sensor_macs
    with model_lock:

        conn = sqlite3.connect("training_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT persistent_id, sensor_mac, rssi, x, y, z FROM training_data WHERE rssi IS NOT NULL")
        data = cursor.fetchall()
        conn.close()

        if len(data) < 3:
            print("⚠️ Not enough valid training data available. Skipping training.")
            # Optionally, you can initialize a basic label_encoder with just 'generic' to avoid errors later:
            label_encoder = LabelEncoder()
            label_encoder.fit(["generic"])
            return None

        sensor_macs = sorted(sensor_data.keys())

        # Gather all persistent_ids
        devices_in_data = [row[0] for row in data]

        # Build feature rows and labels
        features = []
        labels = []
        for persistent_id, sensor_mac, rssi, x, y, z in data:
            row_features = []
            for mac in sensor_macs:
                # Use the RSSI if available; otherwise default to -100
                sensor_rssi = rssi if sensor_mac == mac else -100
                # Retrieve the fixed sensor position from sensor_data
                sensor_pos = sensor_data.get(mac, {}).get("position", (0,0,0))
                # Append both the RSSI and sensor position as part of the feature
                row_features.extend([sensor_rssi, sensor_pos[0], sensor_pos[1], sensor_pos[2]])
            features.append(row_features)
            labels.append([x, y, z])

        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # 1️⃣ Always include "generic" in the label set
        all_labels = set(devices_in_data)
        all_labels.add("generic")
        all_labels = sorted(all_labels)

        # 2️⃣ Fit the label encoder on all_labels
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)

        # 3️⃣ Transform the actual device IDs from the training data
        encoded_devices = label_encoder.transform(devices_in_data)

        # 4️⃣ Combine sensor RSSI features with the encoded device ID
        features = np.column_stack((features, encoded_devices))

        # 5️⃣ Train the RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, labels)

        print(f"✅ ML model trained on {len(features)} samples across multiple devices.")

        # Save the model, sensor_macs, and label_encoder
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, sensor_macs, label_encoder), f)
        print(f"💾 ML Model saved to {MODEL_FILE}")

        return model, sensor_macs, label_encoder

def generate_room_color(room_name):
    """Generate a stable color for a room based on its name."""
    hash_val = int(hashlib.md5(room_name.encode()).hexdigest(), 16)  # Convert room name to a hash
    r = (hash_val % 256) / 255  # Extract Red component
    g = ((hash_val >> 8) % 256) / 255  # Extract Green component
    b = ((hash_val >> 16) % 256) / 255  # Extract Blue component
    return (r, g, b, 1)  # Use alpha=0.5 for transparency

# Trilateration function
def trilaterate(sensors, distances):
    def residuals(pos, sensors, distances):
        return [np.linalg.norm(pos - np.array(sensor)) - dist for sensor, dist in zip(sensors, distances)]

    # Debugging: Print input values before running least squares
    print(f"📡 Trilateration Sensors: {sensors}")
    print(f"📏 Trilateration Distances: {distances}")

    # Initial guess (center of all sensors)
    initial_guess = np.mean(sensors, axis=0)

    try:
        result = least_squares(residuals, initial_guess, args=(sensors, distances))
        print(f"✅ Trilateration Result: {result.x}")
        return result.x
    except Exception as e:
        print(f"❌ Trilateration Error: {e}")
        return None

def get_live_position():
    global rssi_data, model, label_encoder, sensor_macs

    if not rssi_data:
        print("⚠️ No RSSI data available.")
        return None

    persistent_id = next((pid for pid in rssi_data.keys() if len(rssi_data[pid]) >= 3), None)
    if not persistent_id:
        print("⚠️ No valid persistent_id with sufficient RSSI readings.")
        return None

    print(f"🟢 Computing position for {persistent_id} using {len(rssi_data[persistent_id])} RSSI readings...")
    for mac, rssi in rssi_data[persistent_id].items():
        print(f"📡 {mac} → RSSI: {rssi}")

    # Ensure our model is loaded/trained; if not, retrain it.
    if model is None:
        print("⚠️ No trained ML model available. Retraining now...")
        train_ml_model()

    # Build the feature vector from the live sensor data (for all sensors in sensor_macs)
    if model is not None and sensor_macs is not None and label_encoder is not None:
        print("✅ Using trained ML model.")
        rssi_features = []
        for mac in sensor_macs:  # sensor_macs is set as sorted(sensor_data.keys()) during training
            rssi_val = rssi_data[persistent_id].get(mac, -100)
            sensor_pos = sensor_data.get(mac, {}).get("position", (0, 0, 0))
            rssi_features.extend([rssi_val, sensor_pos[0], sensor_pos[1], sensor_pos[2]])

        try:
            encoded_device = label_encoder.transform([persistent_id])[0]
        except (ValueError, AttributeError):
            print(f"⚠️ Persistent ID {persistent_id} not recognized. Falling back to 'generic'.")
            encoded_device = label_encoder.transform(["generic"])[0]

        input_vector = np.append(rssi_features, encoded_device).reshape(1, -1)

        # Get the ML model prediction
        with model_lock:
            if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
                print("⚠️ Model not ready, skipping prediction.")
                ml_prediction = None
            else:
                ml_prediction = model.predict(input_vector)[0]
        print(f"🔍 ML Predicted Position for {persistent_id} → {ml_prediction}")

    else:
        print("ML model or label_encoder not available. Falling back to trilateration.")
        ml_prediction = None

    # Compute a live trilateration estimate using current sensor positions and distances
    sensors = []
    distances = []
    for mac, rssi in rssi_data[persistent_id].items():
        if mac in sensor_data:
            distance = rssi_to_distance(rssi)
            sensors.append(sensor_data[mac]["position"])
            distances.append(distance)
    if len(sensors) >= 3:
        sensors = np.array(sensors)
        distances = np.array(distances)
        trilat_prediction = trilaterate(sensors, distances)
        print(f"📍 Trilateration Prediction: {trilat_prediction}")
    else:
        print("❌ Not enough sensor readings for trilateration.")
        trilat_prediction = None

    # Blend the two estimates.
    # Here we use a simple average if both are available.
    if ml_prediction is not None and trilat_prediction is not None:
        # Compute an average RSSI value across sensors (using sensor_macs)
        avg_rssi = np.mean([rssi_data[persistent_id].get(mac, 100) for mac in sensor_macs])
        # Lower RSSI (lower number) is more reliable. For example, if avg_rssi <= 3, use more weight for trilateration.
        if avg_rssi <= 3:
            w_trilat = 0.7
        else:
            w_trilat = 0.3
        w_ml = 1 - w_trilat
        final_prediction = w_ml * ml_prediction + w_trilat * trilat_prediction
        print(f"🔀 Blended Prediction (avg RSSI: {avg_rssi:.2f}, weights ML: {w_ml:.2f}, trilat: {w_trilat:.2f}): {final_prediction}")
        return final_prediction
    elif ml_prediction is not None:
        return ml_prediction
    elif trilat_prediction is not None:
        return trilat_prediction
    else:
        print("❌ Could not produce any estimate.")
        return None

def store_training_data(persistent_id, mac_address, estimated_position):
    """Stores estimated positions for ML training dynamically."""
    conn = sqlite3.connect("training_data.db")
    cursor = conn.cursor()

    inserted_count = 0  # Count how many records are inserted

    for sensor_mac, rssi in rssi_data.get(persistent_id, {}).items():
        if rssi is None or not isinstance(rssi, (int, float)):
            print(f"⚠️ Skipping sensor {sensor_mac} due to invalid RSSI value: {rssi}")
            continue

        print(f"✅ Storing training data: Persistent ID={persistent_id}, Sensor={sensor_mac}, RSSI={rssi}, Position={estimated_position}")

        cursor.execute("""
        INSERT INTO training_data (persistent_id, mac_address, sensor_mac, rssi, x, y, z)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (persistent_id, mac_address, sensor_mac, float(rssi), estimated_position[0], estimated_position[1], estimated_position[2]))

        inserted_count += 1  # Count successful insertions

    conn.commit()
    conn.close()

    if inserted_count == 0:
        print("⚠️ No training data was inserted! Check if RSSI data exists.")
    else:
        print(f"✅ Successfully stored {inserted_count} new training samples!")

    print("\n🔄 Retraining ML Model with Updated Data...")
    threading.Thread(target=train_ml_model, daemon=True).start()

def on_message(client, userdata, msg):
    global rssi_data
    try:
        payload = json.loads(msg.payload.decode())
        topic_parts = msg.topic.split("/")

        if len(topic_parts) < 5:
            print(f"Skipping malformed topic: {msg.topic}")
            return

        sensor_mac = topic_parts[3].lower()
        rssi_value = payload.get("rssi")
        if rssi_value is None or not isinstance(rssi_value, (int, float)):
            print(f"⚠️ Received invalid RSSI value: {rssi_value}. Skipping update.")
            return

        persistent_id = payload.get("persistent_id")
        mac_address = payload.get("device")

        if persistent_id is None:
            print(f"⚠️ Warning: persistent_id is missing! Using mac_address ({mac_address}) as fallback.")
            persistent_id = mac_address

        if sensor_mac in sensor_data:
            sensor_name = sensor_data[sensor_mac]["name"]

            if persistent_id not in rssi_data:
                rssi_data[persistent_id] = {}

            rssi_data[persistent_id][sensor_mac] = rssi_value  # Store the raw RSSI value
            calculated_distance = rssi_to_distance(rssi_value)
            print(f"Updated rssi_data: {persistent_id} ({sensor_name}) -> {calculated_distance:.2f} m")

            # Ensure we have enough valid readings
#            if len(rssi_data[persistent_id]) >= min(3, len(sensor_data)):  # Adjust dynamically based on available scanners
            if len(rssi_data[persistent_id]) >= 3:
                print("🟢 We have at least 3 RSSI readings, calling get_live_position() now...")
                estimated_position = get_live_position()

                if estimated_position is not None:
                    print(f"🔴 New estimated position: {estimated_position}")

                # If we are in training mode, have not yet stored data, and have an actual position, do so now:
                if (global_state.get("training_mode")
                    and not global_state.get("training_data_stored")
                    and "actual_position" in global_state):
                    print(f"📝 Training mode: Storing actual position {global_state['actual_position']} "
                          f"for persistent_id={persistent_id}")

                    print("🟢 Store new entry to database, calling store_training_data() now...")
                    store_training_data(
                        persistent_id,
                        mac_address,
                        global_state["actual_position"]
                    )

#                    global_state["training_data_stored"] = True
                    print("✅ Done storing training data!")

        else:
            print(f"⚠️ Received RSSI but sensor {sensor_mac} not in known sensors list.")

    except Exception as e:
        print(f"Error processing MQTT message: {e}")

# **MQTT Authentication & Connection**
def connect_mqtt():
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  # Set authentication
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    return client

# Start MQTT Listener
def start_mqtt_listener(persistent_id):
    client = connect_mqtt()
    topic = f"bermuda/{persistent_id}/scanner/+/rssi"
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

def plot_live_trilateration(mac_address):
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        if len(rssi_data) >= 3:
            sensors, distances = zip(*[(sensor_data[mac]["position"], dist) for mac, dist in rssi_data.get(persistent_id, {}).items() if mac in sensor_data])
            #sensors, distances = zip(*[(sensor_data[mac]["position"], dist) for mac, dist in rssi_data.items() if mac in sensor_data])
            estimated_position = trilaterate(np.array(sensors), np.array(distances))

            ax.clear()
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')

            # Plot sensors
            for mac, info in sensor_data.items():
                pos = info["position"]
                ax.scatter(*pos, c='blue', marker='o', s=80, label=info["name"] if mac == list(sensor_data.keys())[0] else "")
                ax.text(pos[0], pos[1], pos[2], info["name"], color='black', fontsize=8)

            # Plot estimated position
            ax.scatter(*estimated_position, c='red', marker='x', s=100, label="Estimated Position")
            ax.text(estimated_position[0], estimated_position[1], estimated_position[2], f"{mac_address}", color='red', fontsize=10)

            plt.legend()
            plt.draw()
            plt.pause(2)  # Refresh every 2 seconds

        time.sleep(1)  # Wait before updating

# Visualization function
def plot_3d_position(sensors, estimated_position):
    plt.ion()  # Enable interactive mode for live updates

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    print("🎥 plot_3d_position() has started... waiting for updates...")

    while True:  # Continuous update loop
        ax.clear()  # Clear previous frame

        # **Get live estimated position from MQTT**
        print("🟢 Calling get_live_position() to fetch latest estimated position...")
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

        # **Plot dashed lines connecting sensors with active RSSI reading to the estimated position**
        persistent_id = global_state.get("persistent_id")
        if persistent_id in rssi_data:
            for mac, rssi in rssi_data[persistent_id].items():
                if mac in sensor_data:
                    sensor_pos = sensor_data[mac]["position"]
                    # Create an array of two points: the sensor and the estimated position
                    line_points = np.array([sensor_pos, estimated_position])
                    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                            linestyle='dashed', color='gray')
                    # Compute the midpoint of the line
                    midpoint = (np.array(sensor_pos) + np.array(estimated_position)) / 2
                    # Display the RSSI value at the midpoint
                    ax.text(midpoint[0], midpoint[1], midpoint[2], f"RSSI: {rssi:.2f}",
                            color='blue', fontsize=8)

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
        print(f"MAC address {mac_address} not found in dataset.")
        return None

    device_data = data[mac_address].get('scanners', {})
    readings = []
    sensor_names = []

    for sensor_mac, sensor_info in device_data.items():
        source_address = sensor_info.get('source', '').lower()  # Normalize source MAC address
        rssi = sensor_info.get('rssi')
        if rssi is not None and source_address in sensor_data:
            calculated_distance = rssi_to_distance(rssi)
            readings.append((sensor_data[source_address]["position"], calculated_distance))
            sensor_names.append(sensor_data[source_address]["name"])

    if len(readings) < 3:
        print(f"Not enough valid sensor readings for MAC {mac_address} to perform trilateration.")
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
    mqtt_thread = threading.Thread(target=start_mqtt_listener, args=(args.persistent_id,))
    mqtt_thread.daemon = True
    mqtt_thread.start()

    # Start 3D visualization with live updates
    plot_3d_position(list(sensor_data.values()), None)
