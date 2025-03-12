# positioning.py
import numpy as np
from scipy.optimize import least_squares
from config import STRONG_RSSI_THRESHOLD, WEAK_RSSI_THRESHOLD, sensor_data, rssi_data, global_state
from utils import rssi_to_distance
import ml_model  # Import the entire module to access its current attributes

def trilaterate(sensors, distances, sensor_weights):
    def residuals(pos, sensors, distances, sensor_weights):
        return [(np.linalg.norm(pos - np.array(sensor)) - dist) * weight
                for sensor, dist, weight in zip(sensors, distances, sensor_weights)]
    initial_guess = np.mean(sensors, axis=0)
    try:
        result = least_squares(residuals, initial_guess, args=(sensors, distances, sensor_weights))
        return result.x
    except Exception as e:
        print(f"Trilateration error: {e}")
        return None

def get_live_position():
    if not rssi_data:
        print("No RSSI data available.")
        return None

    persistent_id = next((pid for pid, readings in rssi_data.items() if len(readings) >= 3), None)
    if not persistent_id:
        print("No valid persistent_id with sufficient RSSI readings.")
        return None

    # Compute sensor weights based on RSSI strength.
    sensor_weights = {}
    for mac in sensor_data:
        rssi = rssi_data[persistent_id].get(mac, -100)
        if rssi >= STRONG_RSSI_THRESHOLD:
            sensor_weights[mac] = 1.5
        elif rssi < WEAK_RSSI_THRESHOLD:
            sensor_weights[mac] = 0.3
        else:
            sensor_weights[mac] = 1.0

    # Ensure the ML model is loaded/trained.
    if ml_model.model is None:
        print("No trained ML model. Training now...")
        ml_model.train_ml_model(rssi_data)

    # Build the feature vector for the ML model.
    sensor_macs = sorted(sensor_data.keys())
    rssi_features = []
    for mac in sensor_macs:
        rssi = rssi_data[persistent_id].get(mac, -100)
        sensor_pos = sensor_data.get(mac, {}).get("position", (0, 0, 0))
        weight = sensor_weights.get(mac, 1.0)
        rssi_features.extend([rssi * weight, sensor_pos[0], sensor_pos[1], sensor_pos[2]])
    for i in range(len(sensor_macs)):
        for j in range(i + 1, len(sensor_macs)):
            rssi_i = rssi_data[persistent_id].get(sensor_macs[i], -100)
            rssi_j = rssi_data[persistent_id].get(sensor_macs[j], -100)
            weight_i = sensor_weights.get(sensor_macs[i], 1.0)
            weight_j = sensor_weights.get(sensor_macs[j], 1.0)
            rssi_features.append((rssi_i - rssi_j) * min(weight_i, weight_j))

    # Ensure label_encoder is initialized.
    if ml_model.label_encoder is None:
        from sklearn.preprocessing import LabelEncoder
        ml_model.label_encoder = LabelEncoder()
        ml_model.label_encoder.fit(["generic"])
    try:
        encoded_device = ml_model.label_encoder.transform([persistent_id])[0]
    except Exception:
        encoded_device = ml_model.label_encoder.transform(["generic"])[0]

    input_vector = np.append(rssi_features, encoded_device).reshape(1, -1)
    with ml_model.model_lock:
        if not hasattr(ml_model.model, 'estimators_') or len(ml_model.model.estimators_) == 0:
            ml_prediction = None
        else:
            ml_prediction = ml_model.model.predict(input_vector)[0]

    # Compute a trilateration estimate.
    sensors_list = []
    distances_list = []
    sensor_weight_list = []
    for mac, rssi in rssi_data[persistent_id].items():
        if mac in sensor_data:
            distance = rssi_to_distance(rssi)
            weight = sensor_weights.get(mac, 1.0)
            sensors_list.append(sensor_data[mac]["position"])
            distances_list.append(distance * weight)
            sensor_weight_list.append(weight)
    if len(sensors_list) >= 3:
        sensors_arr = np.array(sensors_list)
        distances_arr = np.array(distances_list)
        trilat_prediction = trilaterate(sensors_arr, distances_arr, sensor_weight_list)
    else:
        trilat_prediction = None

    # Blend ML and trilateration estimates.
    if ml_prediction is not None and trilat_prediction is not None:
        w_trilat = 0.3  # Adjust weights as needed
        w_ml = 1 - w_trilat
        final_prediction = w_ml * ml_prediction + w_trilat * trilat_prediction
        return final_prediction
    elif ml_prediction is not None:
        return ml_prediction
    elif trilat_prediction is not None:
        return trilat_prediction
    else:
        print("Could not produce any estimate.")
        return None

def get_device_position(yaml_file, mac_address):
    from utils import load_yaml, rssi_to_distance
    data = load_yaml(yaml_file)
    mac_address = mac_address.lower()
    if mac_address not in data:
        print(f"MAC {mac_address} not found in dataset.")
        return None
    device_data = data[mac_address].get('scanners', {})
    readings = []
    for sensor_mac, sensor_info in device_data.items():
        source_address = sensor_info.get('source', '').lower()
        rssi = sensor_info.get('rssi')
        if rssi is not None:
            sensor_pos = sensor_data.get(source_address, {}).get("position", (0, 0, 0))
            readings.append((sensor_pos, rssi_to_distance(rssi)))
    if len(readings) < 3:
        print("Not enough sensor readings for trilateration.")
        return None
    sensors, distances = zip(*readings)
    estimated_position = trilaterate(np.array(sensors), np.array(distances), [1] * len(sensors))
    return estimated_position
