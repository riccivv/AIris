import time
import math
import json
import os
from collections import deque


class LocationTracker:
    """Base location tracker using dead reckoning."""
    
    def __init__(self):
        self.position = {'x': 0, 'y': 0, 'z': 0}  # x,y in meters, z for floor
        self.heading = 0  # Degrees, 0 = North
        self.step_count = 0
        self.travel_history = deque(maxlen=100)
        self.last_update = time.time()
        self.total_distance = 0
        self.floor = 0
        
        # Known landmarks (expandable)
        self.landmarks = [
            {"name": "Starting Point", "x": 0, "y": 0, "floor": 0, "description": "Main entrance"},
            {"name": "Main Cafeteria", "x": 50, "y": 20, "floor": 0, "description": "Ground Floor, East Wing"},
            {"name": "Central Library", "x": 500, "y": 100, "floor": 1, "description": "2nd Floor, Main Building"},
            {"name": "Restroom", "x": 30, "y": 15, "floor": 0, "description": "Ground Floor, West Corridor"},
        ]
        
        # Load saved location
        self.load_location()
    
    def update_position(self, distance_moved, heading=None):
        """Update position based on distance and heading."""
        if heading is not None:
            self.heading = heading
        
        heading_rad = math.radians(self.heading)
        self.position['x'] += distance_moved * math.sin(heading_rad)
        self.position['y'] += distance_moved * math.cos(heading_rad)
        self.total_distance += distance_moved
        self.step_count += 1
        
        self.travel_history.append({
            'position': self.position.copy(),
            'time': time.time(),
            'heading': self.heading
        })
        
        self.last_update = time.time()
        
        # Auto-save periodically
        if self.step_count % 10 == 0:
            self.save_location()
    
    def update_from_gps(self, lat, lon, altitude=None):
        """Update position from GPS coordinates."""
        # Convert GPS to local coordinates (simplified)
        # This would need proper coordinate conversion in real implementation
        self.position['x'] = lat
        self.position['y'] = lon
        if altitude is not None:
            self.position['z'] = altitude
        
        self.last_update = time.time()
        self.save_location()
    
    def get_position(self):
        """Get current position."""
        return self.position.copy()
    
    def get_distance_from_start(self):
        """Calculate 2D distance from start."""
        return math.sqrt(self.position['x']**2 + self.position['y']**2)
    
    def get_nearest_landmark(self, max_distance=100):
        """Find nearest landmark within max_distance."""
        nearest = []
        
        for landmark in self.landmarks:
            distance = math.sqrt(
                (self.position['x'] - landmark['x'])**2 + 
                (self.position['y'] - landmark['y'])**2
            )
            
            if distance <= max_distance:
                landmark_copy = landmark.copy()
                landmark_copy['distance'] = distance
                nearest.append(landmark_copy)
        
        # Sort by distance
        nearest.sort(key=lambda x: x['distance'])
        return nearest[0] if nearest else None
    
    def get_all_nearby_landmarks(self, max_distance=100):
        """Get all landmarks within max_distance."""
        return sorted(
            [
                {**landmark, 'distance': math.sqrt(
                    (self.position['x'] - landmark['x'])**2 + 
                    (self.position['y'] - landmark['y'])**2
                )}
                for landmark in self.landmarks
                if math.sqrt(
                    (self.position['x'] - landmark['x'])**2 + 
                    (self.position['y'] - landmark['y'])**2
                ) <= max_distance
            ],
            key=lambda x: x['distance']
        )
    
    def get_location_description(self):
        """Generate human-readable location description."""
        distance_from_start = self.get_distance_from_start()
        nearest = self.get_nearest_landmark()
        
        parts = []
        parts.append(f"You are approximately {distance_from_start:.1f} meters from your starting point")
        
        if nearest and nearest['distance'] < 100:
            parts.append(f"near {nearest['name']}")
            if nearest['distance'] > 5:
                parts.append(f"({nearest['distance']:.1f} meters away)")
        
        if self.floor > 0:
            parts.append(f"on floor {self.floor}")
        
        parts.append(f"You have traveled {self.total_distance:.1f} meters total")
        
        return " ".join(parts) + "."
    
    def reset(self):
        """Reset tracker."""
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.heading = 0
        self.step_count = 0
        self.total_distance = 0
        self.travel_history.clear()
        self.save_location()
    
    def save_location(self, filename='location_data.json'):
        """Save current location to file."""
        try:
            data = {
                'position': self.position,
                'heading': self.heading,
                'total_distance': self.total_distance,
                'floor': self.floor,
                'timestamp': time.time()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Location Save Error]: {e}")
    
    def load_location(self, filename='location_data.json'):
        """Load saved location."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.position = data.get('position', {'x': 0, 'y': 0, 'z': 0})
                    self.heading = data.get('heading', 0)
                    self.total_distance = data.get('total_distance', 0)
                    self.floor = data.get('floor', 0)
                    print(f"[Location]: Restored previous location")
        except Exception as e:
            print(f"[Location Load Error]: {e}")