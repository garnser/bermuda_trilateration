# plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import numpy as np
import time
from config import sensor_data, room_data, floor_boundaries, global_state, rssi_data, CEILING_HEIGHT, global_state
from utils import remove_overlapping_ceiling, find_room, generate_room_color

# Pre-generate room colors.
room_colors = {r["name"]: generate_room_color(r["name"]) for r in room_data}

def plot_live_trilateration(mac_address):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while True:
        # (Optional) implement live trilateration plotting here.
        plt.pause(1)
        time.sleep(0.5)

def plot_3d_position():
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    from positioning import get_live_position  # Import live updater
    while True:
        ax.clear()
        # Get the live estimated position on every iteration.
        estimated_position = get_live_position()

        floor_heights = {room["floor"]: room["floor"] * CEILING_HEIGHT for room in room_data}
        ceilings = remove_overlapping_ceiling(floor_boundaries)
        custom_handles = []

        # Draw floor boundaries.
        for floor, boundary in floor_boundaries.items():
            floor_z = floor_heights.get(floor, 0)
            ax.plot(boundary[:, 0], boundary[:, 1],
                    np.full_like(boundary[:, 0], floor_z),
                    color='black', linestyle='dashed', label=f'Floor {floor}')

        sensor_position_patch = mpatches.Patch(color='blue', label='Sensor')
        custom_handles.append(sensor_position_patch)

        estimated_position_patch = mpatches.Patch(color='red', label='Estimated Position')
        custom_handles.append(estimated_position_patch)

        house_boundaries_patch = mpatches.Patch(color='black', label='House Boundaries')
        custom_handles.append(house_boundaries_patch)

        for floor, poly in ceilings.items():
            if not poly.is_empty:
                x, y = poly.exterior.xy
                # Create the polygon at the ceiling's Z-level: (floor + 1)*3
                ceiling_corners = [(x[i], y[i], (floor + 1) * CEILING_HEIGHT) for i in range(len(x))]
                ceiling_surface = Poly3DCollection([ceiling_corners], color='gray', alpha=0.3)
                ax.add_collection3d(ceiling_surface)

        ceiling_patch = mpatches.Patch(color='gray', alpha=0.3, label='Ceiling')
        custom_handles.append(ceiling_patch)

        # Draw rooms.
        for room in room_data:
            floor_z = floor_heights[room["floor"]]
            room_corners = [(x, y, floor_z) for x, y in room["corners"]]
            room_surface = Poly3DCollection([room_corners], color=room_colors[room["name"]], alpha=0.5)
            ax.add_collection3d(room_surface)
            custom_handles.append(mpatches.Patch(color=room_colors[room["name"]],
                                                  label=f'{room["name"]} (Floor {room["floor"]})'))

        # Draw sensors.
        for mac, data in sensor_data.items():
            pos = data["position"]
            ax.scatter(*pos, c='blue', marker='o', s=80)
            ax.text(pos[0], pos[1], pos[2], data["name"], fontsize=8)

        # Draw connections from sensors to estimated position.
        persistent_id = global_state.get("persistent_id")
        if persistent_id in rssi_data and estimated_position is not None:
            for mac, rssi in rssi_data[persistent_id].items():
                if mac in sensor_data:
                    sensor_pos = sensor_data[mac]["position"]
                    line = np.array([sensor_pos, estimated_position])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2],
                            linestyle='dashed', color='gray')
                    midpoint = (sensor_pos + estimated_position) / 2
                    if isinstance(rssi, list) and len(rssi) > 0:
                        rssi_val = np.median(rssi)
                    else:
                        rssi_val = -100
                    ax.text(midpoint[0], midpoint[1], midpoint[2],
                            f"RSSI: {rssi_val:.2f} dB", color='blue', fontsize=8)

        # Plot the estimated position.
        if estimated_position is not None:
            ax.scatter(*estimated_position, c='red', marker='x', s=100, label='Estimated Position')

            if global_state.get("training_mode") and global_state.get("actual_position"):
                actual_pos = global_state["actual_position"]
                ax.scatter(
                    actual_pos[0], actual_pos[1], actual_pos[2],
                    c='green', marker='o', s=60, label='Training Ground Truth'
                )

            room_name = find_room(estimated_position, room_data)
            ax.text(estimated_position[0] + 0.3, estimated_position[1], estimated_position[2],
                    room_name, fontsize=10, fontweight='bold', color='red')

            # Add info box with estimated position (2 decimals), room name, and sensor RSSI values.
            pos_str = f"({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f})"
            actual_str = ""
            if global_state.get("training_mode") and global_state.get("actual_position"):
                actual_pos = global_state["actual_position"]
                actual_str = f"Actual Position: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f})\n"

            persistent_id = global_state.get("persistent_id")
            rssi_info = ""
            if persistent_id in rssi_data:
                for mac, rssi in rssi_data[persistent_id].items():
                    sensor_name = sensor_data.get(mac, {}).get("name", mac)
                    if isinstance(rssi, list) and len(rssi) > 0:
                        rssi_val = np.median(rssi)
                    else:
                        rssi_val = -100
                    rssi_info += f"{sensor_name} ({mac}): {rssi_val:.2f} dB\n"

            info_text = f"Estimated Position: {pos_str}\n"
            info_text += actual_str
            info_text += f"Room: {room_name}\n"
            if global_state.get("training_mode") and global_state.get("training_score") is not None:
                info_text += f"Training Score: {global_state['training_score']:.2f}\n"
            info_text += rssi_info

            if "trilat_rms" in global_state and global_state["trilat_rms"] is not None:
                rms_val = global_state["trilat_rms"]
                info_text += f"Trilat RMS: {rms_val:.2f} m\n"

            if not hasattr(plot_3d_position, "info_box"):
                plot_3d_position.info_box = fig.text(0.02, 0.5, info_text, fontsize=10,
                                                      verticalalignment='center', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            else:
                plot_3d_position.info_box.set_text(info_text)

        ax.legend(handles=custom_handles, loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.draw()
        plt.pause(2)
