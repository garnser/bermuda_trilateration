import yaml
import hashlib
from shapely.geometry import Polygon, Point
from config import STRONG_RSSI_THRESHOLD, WEAK_RSSI_THRESHOLD, TX_POWER

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return {k.lower(): v for k, v in data.items()}

def rssi_to_distance(rssi, tx_power=TX_POWER, n=2.0):
    """Convert RSSI to distance (in meters) using a path-loss model."""
    return 10 ** ((tx_power - rssi) / (10 * n))

def generate_room_color(room_name):
    """Generate a stable color for a room based on its name."""
    hash_val = int(hashlib.md5(room_name.encode()).hexdigest(), 16)
    r = (hash_val % 256) / 255
    g = ((hash_val >> 8) % 256) / 255
    b = ((hash_val >> 16) % 256) / 255
    return (r, g, b, 1)

def remove_overlapping_ceiling(floor_boundaries):
    """Remove overlapping ceiling sections based on floor boundaries."""
    ceilings = {}
    for floor, boundary in floor_boundaries.items():
        poly_lower = Polygon(boundary)
        if floor + 1 in floor_boundaries:
            poly_upper = Polygon(floor_boundaries[floor + 1])
            ceilings[floor] = poly_lower.difference(poly_upper)
        else:
            ceilings[floor] = poly_lower
    return ceilings

def find_room(estimated_position, room_data):
    """Determine which room the estimated position is in."""
    estimated_point = Point(estimated_position[:2])
    estimated_floor = int(estimated_position[2] // 3)
    for room in room_data:
        if room["floor"] == estimated_floor:
            room_polygon = Polygon(room["corners"])
            if room_polygon.contains(estimated_point):
                return room["name"]
    return "Unknown Room"
