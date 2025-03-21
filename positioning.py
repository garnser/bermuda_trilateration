# positioning.py
import numpy as np
from logger import logger, training_logger
from scipy.optimize import least_squares
from config import STRONG_RSSI_THRESHOLD, WEAK_RSSI_THRESHOLD, sensor_data, rssi_data, global_state, TX_POWER
from utils import rssi_to_distance, estimate_tx_power
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
        logger.error(f"[ERROR] Trilateration error: {e}")
        return None

def get_live_position():
    if not rssi_data:
        logger.warning("[WARNING] No RSSI data available.")
        return None

    persistent_id = next((pid for pid, readings in rssi_data.items() if len(readings) >= 3), None)
    if not persistent_id:
        logger.warning("[WARNING] No valid persistent_id with sufficient RSSI readings.")
        return None

    # Compute sensor weights based on RSSI strength.
    sensor_weights = {}
    for mac in sensor_data:
        # Now rssi_data[persistent_id][mac] is a list of recent RSSI samples.
        rssi_list = rssi_data[persistent_id].get(mac, [])
        if not rssi_list:
            rssi_val = -100
        else:
            rssi_val = np.median(rssi_list)

        if rssi_val >= STRONG_RSSI_THRESHOLD:
            sensor_weights[mac] = 1.5
        elif rssi_val < WEAK_RSSI_THRESHOLD:
            sensor_weights[mac] = 0.3
        else:
            sensor_weights[mac] = 1.0

    # Ensure the ML model is loaded/trained if training data exists.
    if ml_model.model is None:
        logger.info("[INFO] No trained ML model. Training now...")
        ml_model.train_ml_model(rssi_data)

    # Build the feature vector for the ML model.
    sensor_macs = sorted(sensor_data.keys())
    rssi_features = []
    for mac in sensor_macs:
        # Use the median RSSI from the rolling list
        rssi_list = rssi_data[persistent_id].get(mac, [])
        if not rssi_list:
            rssi_val = -100
        else:
            rssi_val = np.median(rssi_list)
        sensor_pos = sensor_data.get(mac, {}).get("position", (0, 0, 0))
        weight = sensor_weights.get(mac, 1.0)
        rssi_features.extend([rssi_val * weight, sensor_pos[0], sensor_pos[1], sensor_pos[2]])
    for i in range(len(sensor_macs)):
        for j in range(i + 1, len(sensor_macs)):
            # Also for pairwise differences, use the median
            list_i = rssi_data[persistent_id].get(sensor_macs[i], [])
            list_j = rssi_data[persistent_id].get(sensor_macs[j], [])
            if list_i:
                rssi_i = np.median(list_i)
            else:
                rssi_i = -100
            if list_j:
                rssi_j = np.median(list_j)
            else:
                rssi_j = -100
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

    # DEBUG: Log input vector details.
    logger.debug("[DEBUG] Constructed input_vector: %s", input_vector)
    logger.debug("[DEBUG] input_vector shape: %s", input_vector.shape)
    if np.any(np.isnan(input_vector)):
        logger.error("[ERROR] NaN values detected in input_vector!")

    with ml_model.model_lock:
        if ml_model.model is None or not hasattr(ml_model.model, 'estimators_') or len(ml_model.model.estimators_) == 0:
            ml_prediction = None
        else:
            # DEBUG: Get predictions from each estimator.
            individual_preds = [est.predict(input_vector)[0] for est in ml_model.model.estimators_]
            logger.debug("[DEBUG] Individual estimator predictions: %s", individual_preds)
            ml_prediction = np.mean(individual_preds, axis=0)

    # Check for NaN in ML prediction
    if ml_prediction is not None and np.any(np.isnan(ml_prediction)):
        logger.warning("[WARNING] ML prediction contains NaN values. Ignoring ML prediction.")
        ml_prediction = None

    # Compute a trilateration estimate.
    sensors_list = []
    distances_list = []
    sensor_weight_list = []
    for mac, rssi in rssi_data[persistent_id].items():
        if mac in sensor_data:
            # Look up the newly learned sensor-specific tx_power; fallback to config.TX_POWER
            sensor_tx_power = sensor_data[mac].get("tx_power", TX_POWER)
            # Use median again for the final distance calculation
            rssi_list = rssi_data[persistent_id].get(mac, [])
            if not rssi_list:
                rssi_val = -100
            else:
                rssi_val = np.median(rssi_list)
            distance = rssi_to_distance(rssi_val, tx_power=sensor_tx_power)
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
    elif ml_prediction is not None:
        final_prediction = ml_prediction
    elif trilat_prediction is not None:
        final_prediction = trilat_prediction
    else:
        logger.warning("[ERROR] Could not produce any estimate.")
        return None

    # Verify that the final prediction does not contain NaN values.
    if np.any(np.isnan(final_prediction)):
        logger.error("[ERROR] Final prediction contains NaN values. Returning fallback position.")
        return None

    # Apply filtering placeholder to the final prediction.
    from utils import apply_filtering
    final_prediction = apply_filtering(final_prediction)

    # Training mode: compute score only if actual position is provided.
    if global_state.get("training_mode") and global_state.get("actual_position"):
        actual = np.array(global_state["actual_position"])
        estimated = np.array(final_prediction)
        error = np.linalg.norm(actual - estimated)
        score = 1 - min(1, error / 5.0)
        training_logger.info(f"[TRAINING] Actual: {actual}, Estimated: {estimated}, Error: {error:.2f}, Score: {score:.2f}")
        global_state["training_score"] = score

    if 'last_position' not in global_state:
        global_state['last_position'] = final_prediction
    alpha = 0.5  # smoothing factor
    smoothed_position = alpha * final_prediction + (1 - alpha) * global_state['last_position']
    global_state['last_position'] = smoothed_position

    return smoothed_position
#    return final_prediction

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
