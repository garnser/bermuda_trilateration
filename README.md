# Trilateration with ML Training
This project estimates the position of a mobile device using a combination of **RSSI-based trilateration** and a **machine learning model** (Random Forest). The system collects BLE RSSI readings from multiple sensors, processes them through both a classical trilateration algorithm and an ML regression model, then combines both estimates to output a final (𝑥,𝑦,𝑧) position.

## Table of Contents
1. Features
2. Architecture Overview
3. Installation & Setup
4. Project Structure
5. Usage
6. Training Mode & 1:Many Data Collection
7. Visualization
8. Extending the Project

## Features
- **MQTT-based data ingestion**: Gathers RSSI readings from BLE sensors.
- **Database persistence**: Saves training data (RSSI readings + ground truth) in a local SQLite database.
- **Machine learning**: Trains a Random Forest regressor on the collected data for improved position accuracy.
- **Classical trilateration**: Combines direct distance-based computation with ML outputs.
3D visualization: Provides a real-time 3D plot of sensor positions, rooms, floors, and the live estimated device location.
- **1:Many data collection**: Optionally stores multiple RSSI snapshots at the same location to account for real-world RSSI fluctuations.

## Architecture Overview
1. **Sensors** broadcast BLE advertisement packets.
2. **MQTT Broker** receives sensor-specific RSSI data for a given device ID (i.e., `persistent_id`).
3. **mqtt_client.py** listens to MQTT messages. Upon receiving enough sensor RSSI readings (≥3), it triggers position estimation.
4. **positioning.py** computes two parallel estimates:
  - Trilateration using distances (derived from RSSI) and least-squares optimization.
  - ML model using a RandomForestRegressor that was trained on historical data.
5. **Results** from both methods are merged (with adjustable weighting).
6. `plotting.py` repeatedly plots the new position in a 3D environment, along with rooms, floors, sensors, etc.
7. **Training Data** can be stored by specifying a ground-truth position (`--train`) when running. This saves RSSI + known (𝑥,𝑦,𝑧) to the SQLite database for future model training.

## Installation & Setup
1. Clone or download this repository.
2. **Install Python dependencies** (we recommend Python ≥ 3.7). Example:

  ```bash
  pip install -r requirements.txt
  ```
  The required packages typically include:
  - numpy
  - matplotlib
  - paho-mqtt
  - scipy
  - sklearn
  - shapely (for geometric operations)

3. **Configure MQTT**: In `config.py`, update:
  - `MQTT_BROKER`
  - `MQTT_PORT`
  - `MQTT_USERNAME`
  - `MQTT_PASSWORD` so they match your MQTT broker settings.

4. **Configure Sensor Data**: In `config.py`, each sensor MAC and its known (𝑥,𝑦,𝑧) position is listed under `sensor_data`. Adjust these to match your environment.
5. **Configure Building Geometry**: If desired, modify the `room_data` and `floor_boundaries` to reflect your actual rooms/floors.

## Project Structure
```graphql
.
├─ config.py         # Central config for MQTT, ML model, global state, sensor positions, etc.
├─ database.py       # Initializes SQLite DB and inserts training samples.
├─ logger.py         # Custom logging setup and training logger.
├─ main.py           # Entry point. Sets up DB, loads model, starts MQTT, and starts 3D plotting.
├─ ml_model.py       # RandomForestRegressor training, model loading, and saving.
├─ mqtt_client.py    # Connects/subscribes to MQTT broker, processes incoming RSSI messages.
├─ plotting.py       # Continuously plots the 3D scene (floors, sensors, device position).
├─ positioning.py    # Trilateration plus ML-based position estimation; merges outputs.
├─ utils.py          # Helper utilities (e.g., RSSI→distance, geometric operations, etc.)
├─ requirements.txt  # Python dependencies (if provided).
└─ training_data.db  # SQLite DB (created at runtime) with training samples.
```

### Key Files
- `main.py`

  Entry point. Parse CLI arguments (the device’s persistent ID and optional training coordinates), then launches MQTT listener and the live 3D plot.

- `mqtt_client.py`

  Subscribes to MQTT topics, extracts RSSI readings, and triggers position estimation. If in training mode, also inserts data into `training_data.db`.

- `ml_model.py`
  
  Trains and saves the RandomForest model to `ml_model.pkl`. Uses data from `training_data.db`.

- `positioning.py`
  
  Combines raw trilateration (via `scipy.optimize.least_squares`) with the ML model’s predictions into a final (𝑥,𝑦,𝑧).

- `plotting.py`

  Draws a 3D view of sensors, rooms/floors, and the current estimated device location in real time.

## Usage
1. **Basic usage** (no training):
```bash
python main.py <persistent_id>
```
  - `<persistent_id>` is a string/device identifier that helps track multiple sessions/devices.
2. **Training Mode** :
```bash
python main.py <persistent_id> --train <x> <y> <z>
```
  - Example:
    ```bash
    python main.py phone123 --train 3.2 4.7 1.0
    ```
  - This instructs the system that the “phone123” device is currently at (3.2,4.7,1.0).
  - RSSI data will be stored in the DB with that ground-truth coordinate.
3. **Check logs**:

   Logging is set to `DEBUG` by default (see `LOGLEVEL` in `config.py`). Adjust or watch the console for debug/training logs.

4. **Exiting**:
Close the plot window or hit Ctrl+C to stop the script.

## Training Mode & 1:Many Data Collection
When training mode is active (`--train X Y Z`), the system stores **RSSI + ground truth** in the `training_data` table.

- **1:Many approach**: In `mqtt_client.py`, each time enough RSSI data is received, a new row is stored—even if the device has not moved—leading to multiple snapshots for the same 
(𝑥,𝑦,𝑧).
- The key variable `global_state["last_store_time"]` throttles how frequently new rows are stored. Adjust that logic to collect more (or fewer) samples.

To **retrain** the model:

1. Gather training data for different positions (the more variety, the better).
2. Then run the system again (or send enough MQTT data). The code checks if a trained model exists; if not found (or if forced), it calls `train_ml_model` in `ml_model.py` to read from `training_data.db` and build a new RandomForest.

## Visualization
`plot_3d_position()` in `plotting.py` displays a live 3D scene of:

- Floor boundaries (and optional ceilings).
- Rooms (colored polygons).
- Sensor locations (blue dots).
- The device’s estimated position (red “X”).
- Lines from each sensor to the device, labeled with RSSI.

Position is updated periodically (default every 2 seconds). Pressing Ctrl+C or closing the matplotlib window stops the script.

## Extending the Project
- **Filtering**: Implement a Kalman or Particle Filter in `apply_filtering(position)` in `utils.py` to smooth position jumps.
- **Custom weighting**: Tweak the `w_ml` vs. `w_trilat` ratio in `positioning.py` to give more or less emphasis to the ML model vs. classical trilateration.
- **Better training logic**: Instead of a time-based threshold, you could store a new sample if the RSSI distribution changes or after a certain number of packets.
- **Additional features**: Consider including orientation data, device motion tracking, or multi-floor constraints to refine your estimates.


Happy positioning! If you have questions or improvements, feel free to open issues or pull requests.