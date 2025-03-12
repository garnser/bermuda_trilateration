# main.py
import argparse
import threading
from database import init_database
from ml_model import load_ml_model
from mqtt_client import start_mqtt_listener
from plotting import plot_3d_position
from config import global_state

def main():
    parser = argparse.ArgumentParser(description="Trilateration with ML training")
    parser.add_argument("persistent_id", help="Persistent ID of the device")
    parser.add_argument("--train", nargs=3, type=float, help="Provide actual X, Y, Z position for training")
    args = parser.parse_args()

    init_database()
    load_ml_model()
    global_state["training_mode"] = args.train is not None
    global_state["persistent_id"] = args.persistent_id
    if global_state["training_mode"]:
        global_state["actual_position"] = tuple(args.train)

    # Start the MQTT listener in a separate thread.
    mqtt_thread = threading.Thread(target=start_mqtt_listener, args=(args.persistent_id,))
    mqtt_thread.daemon = True
    mqtt_thread.start()

    # Start 3D visualization (parameters can be adjusted as needed).
    plot_3d_position()

if __name__ == "__main__":
    main()
