import time
import math
import json
import os
from collections import deque
from ..config import DATA_DIR


class LocationTracker:
    """Enhanced location tracker for simulated (dead-reckoning) positioning.

    Tracks the user's position in 2D/3D based on walked distance and heading.
    Used to report location, find nearby landmarks, and drive navigation
    distances. The GPS/phone source was removed; this is a purely local grid.
    """

    def __init__(self):
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.heading = 0  # Degrees, 0 = North, 90 = East
        self.step_count = 0
        self.travel_history = deque(maxlen=100)
        self.last_update = time.time()
        self.total_distance = 0
        self.floor = 0
        self.current_speed = 0

        # Known landmarks on the local grid
        self.landmarks = [
            {
                "name": "Starting Point", 
                "x": 0, "y": 0, "floor": 0, 
                "description": "Main entrance",
                "keywords": ["start", "entrance", "beginning", "origin"]
            },
            {
                "name": "Main Cafeteria", 
                "x": 50, "y": 20, "floor": 0, 
                "description": "Ground Floor, East Wing",
                "keywords": ["cafeteria", "canteen", "dining", "food", "eat", "lunch"]
            },
            {
                "name": "Central Library", 
                "x": 500, "y": 100, "floor": 1, 
                "description": "2nd Floor, Main Building",
                "keywords": ["library", "study", "books", "reading"]
            },
            {
                "name": "Restroom", 
                "x": 30, "y": 15, "floor": 0, 
                "description": "Ground Floor, West Corridor",
                "keywords": ["restroom", "toilet", "bathroom", "washroom"]
            },
        ]

        # Current destination tracking
        self.current_destination = None
        self.arrived_at_destination = False

        # Load saved location
        self.load_location()

    def update_position(self, distance_moved, heading=None):
        """Update position based on distance and heading (dead reckoning)."""
        if heading is not None:
            self.heading = heading

        # Convert heading to radians (0=North, 90=East)
        heading_rad = math.radians(self.heading)

        # Update X (East-West) and Y (North-South)
        self.position['x'] += distance_moved * math.sin(heading_rad)
        self.position['y'] += distance_moved * math.cos(heading_rad)

        # Update Z (total distance traveled)
        self.position['z'] += distance_moved
        self.total_distance += distance_moved

        # Update speed
        time_diff = time.time() - self.last_update
        if time_diff > 0:
            self.current_speed = distance_moved / time_diff

        self.step_count += 1
        self.last_update = time.time()

        # Record history
        self.travel_history.append({
            'position': self.position.copy(),
            'time': time.time(),
            'heading': self.heading,
            'speed': self.current_speed
        })

        # Check if arrived at destination
        if self.current_destination:
            distance_to_dest = self.get_distance_to(self.current_destination)
            if distance_to_dest < 3.0:  # Within 3 meters
                self.arrived_at_destination = True

        # Auto-save periodically
        if self.step_count % 10 == 0:
            self.save_location()

    def get_distance_to(self, target):
        """Calculate distance to a target location."""
        # Handle dictionary input
        if isinstance(target, dict):
            if 'x' in target and 'y' in target:
                target_x = target['x']
                target_y = target['y']
            else:
                return 0
        # Handle tuple/list input
        elif isinstance(target, (tuple, list)) and len(target) >= 2:
            target_x = target[0]
            target_y = target[1]
        else:
            return 0

        dx = target_x - self.position['x']
        dy = target_y - self.position['y']
        return math.sqrt(dx**2 + dy**2)

    def get_direction_to(self, target):
        """Get direction to target (compass direction)."""
        # Handle dictionary input
        if isinstance(target, dict):
            if 'x' in target and 'y' in target:
                target_x = target['x']
                target_y = target['y']
            else:
                # Try to find landmark by name
                if 'name' in target:
                    landmark = self.find_landmark(target['name'])
                    if landmark:
                        target_x = landmark['x']
                        target_y = landmark['y']
                    else:
                        return {'angle': 0, 'direction': 'North', 'distance': 0}
                else:
                    return {'angle': 0, 'direction': 'North', 'distance': 0}
        # Handle tuple/list input
        elif isinstance(target, (tuple, list)) and len(target) >= 2:
            target_x = target[0]
            target_y = target[1]
        else:
            return {'angle': 0, 'direction': 'North', 'distance': 0}

        # Calculate differences
        dx = target_x - self.position['x']
        dy = target_y - self.position['y']

        # Calculate angle (0 = North, clockwise)
        angle = math.degrees(math.atan2(dx, dy))
        if angle < 0:
            angle += 360

        # Convert to compass direction
        directions = ['North', 'North-East', 'East', 'South-East', 
                     'South', 'South-West', 'West', 'North-West']
        index = round(angle / 45) % 8

        return {
            'angle': angle,
            'direction': directions[index],
            'distance': math.sqrt(dx**2 + dy**2)
        }

    def is_at_location(self, landmark_name, threshold=5.0):
        """Check if user is at a specific location."""
        landmark = self.find_landmark(landmark_name)
        if landmark:
            distance = self.get_distance_to(landmark)
            return distance < threshold
        return False

    def find_landmark(self, name_or_keyword):
        """Find landmark by name or keyword."""
        search_term = name_or_keyword.lower()

        # Try exact name match first
        for landmark in self.landmarks:
            if landmark['name'].lower() == search_term:
                return landmark

        # Try keyword match
        for landmark in self.landmarks:
            if any(keyword in search_term for keyword in landmark.get('keywords', [])):
                return landmark

        # Try partial name match
        for landmark in self.landmarks:
            if search_term in landmark['name'].lower():
                return landmark

        return None

    def get_nearest_landmark(self, max_distance=100):
        """Find nearest landmark within max_distance."""
        nearest = []

        for landmark in self.landmarks:
            distance = self.get_distance_to(landmark)

            if distance <= max_distance:
                landmark_copy = landmark.copy()
                landmark_copy['distance'] = distance
                nearest.append(landmark_copy)

        # Sort by distance
        nearest.sort(key=lambda x: x['distance'])
        return nearest[0] if nearest else None

    def get_all_nearby_landmarks(self, max_distance=100):
        """Get all landmarks within max_distance."""
        nearby = []

        for landmark in self.landmarks:
            distance = self.get_distance_to(landmark)
            if distance <= max_distance:
                landmark_copy = landmark.copy()
                landmark_copy['distance'] = distance
                nearby.append(landmark_copy)

        nearby.sort(key=lambda x: x['distance'])
        return nearby

    def get_location_description(self):
        """Generate a human-readable location description (dead reckoning)."""
        distance_from_start = self.get_distance_from_start()
        nearest = self.get_nearest_landmark()

        parts = []
        parts.append(f"You are approximately {distance_from_start:.1f} meters from your starting point")

        if nearest and nearest['distance'] < 50:
            parts.append(f"near {nearest['name']}")
            if nearest['distance'] > 3:
                parts.append(f"({nearest['distance']:.1f} meters away)")
            else:
                parts.append("(you are here)")

        if self.floor > 0:
            parts.append(f"on floor {self.floor}")

        parts.append(f"You have traveled {self.total_distance:.1f} meters total")

        return " ".join(parts) + "."

    def get_distance_from_start(self):
        """Calculate 2D distance from start."""
        return math.sqrt(self.position['x']**2 + self.position['y']**2)

    def get_position(self):
        """Get current position."""
        return self.position.copy()

    def reset(self):
        """Reset tracker to starting position."""
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.heading = 0
        self.step_count = 0
        self.total_distance = 0
        self.travel_history.clear()
        self.current_destination = None
        self.arrived_at_destination = False
        self.save_location()

    def save_location(self, filename='location_data.json'):
        """Save current location to file."""
        try:
            filepath = os.path.join(DATA_DIR, filename)
            data = {
                'position': self.position,
                'heading': self.heading,
                'total_distance': self.total_distance,
                'floor': self.floor,
                'timestamp': time.time()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Location Save Error]: {e}")

    def load_location(self, filename='location_data.json'):
        """Load saved location from file."""
        try:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.position = data.get('position', {'x': 0, 'y': 0, 'z': 0})
                    self.heading = data.get('heading', 0)
                    self.total_distance = data.get('total_distance', 0)
                    self.floor = data.get('floor', 0)
                    print(f"[Location]: Restored previous location")
        except Exception as e:
            print(f"[Location Load Error]: {e}")