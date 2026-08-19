import time
import math
from collections import deque
import json
import os


class LocationTracker:
    """Track approximate location using dead reckoning and known landmarks."""
    
    def __init__(self):
        self.position = {'x': 0, 'y': 0}  # Starting position (0,0)
        self.heading = 0  # Degrees, 0 = North
        self.step_count = 0
        self.travel_history = deque(maxlen=100)
        self.last_update = time.time()
        self.total_distance = 0
        
        # Known landmarks (you can expand this)
        self.landmarks = [
            {"name": "Starting Point", "x": 0, "y": 0, "description": "Main entrance"},
            {"name": "Main Cafeteria", "x": 50, "y": 20, "description": "Ground Floor, East Wing"},
            {"name": "Central Library", "x": 500, "y": 100, "description": "2nd Floor, Main Building"},
            {"name": "Restroom", "x": 30, "y": 15, "description": "Ground Floor, West Corridor"},
        ]
        
        # Load saved location if exists
        self.load_location()
    
    def update_heading(self, new_heading):
        """Update current heading direction."""
        self.heading = new_heading
        
    def update_position(self, distance_moved):
        """Update position based on distance moved in current heading."""
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
        
        # Auto-save location periodically
        if self.step_count % 10 == 0:
            self.save_location()
    
    def get_position(self):
        """Get current estimated position."""
        return self.position.copy()
    
    def get_distance_from_start(self):
        """Calculate distance from starting position."""
        return math.sqrt(self.position['x']**2 + self.position['y']**2)
    
    def get_nearest_landmark(self):
        """Find nearest known landmark to current position."""
        nearest = None
        min_distance = float('inf')
        
        for landmark in self.landmarks:
            distance = math.sqrt(
                (self.position['x'] - landmark['x'])**2 + 
                (self.position['y'] - landmark['y'])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest = landmark.copy()
                nearest['distance'] = distance
        
        return nearest
    
    def get_direction_to(self, target_x, target_y):
        """Calculate direction and distance to a target location."""
        dx = target_x - self.position['x']
        dy = target_y - self.position['y']
        
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dx, dy))
        
        # Convert to compass direction
        directions = ['North', 'North-East', 'East', 'South-East', 
                     'South', 'South-West', 'West', 'North-West']
        index = round(angle / 45) % 8
        direction = directions[index]
        
        return {
            'distance': distance,
            'direction': direction,
            'angle': angle
        }
    
    def get_location_description(self):
        """Generate human-readable location description."""
        position = self.get_position()
        distance_from_start = self.get_distance_from_start()
        nearest = self.get_nearest_landmark()
        
        description = f"You are approximately {distance_from_start:.1f} meters from your starting point"
        
        if nearest and nearest['distance'] < 100:
            description += f", near {nearest['name']}"
            if nearest['distance'] > 5:
                description += f" ({nearest['distance']:.1f} meters away)"
        
        description += f". You have traveled {self.total_distance:.1f} meters total."
        
        return description
    
    def reset(self):
        """Reset tracker to starting position."""
        self.position = {'x': 0, 'y': 0}
        self.heading = 0
        self.step_count = 0
        self.total_distance = 0
        self.travel_history.clear()
        self.save_location()
    
    def save_location(self):
        """Save current location to file."""
        try:
            data = {
                'position': self.position,
                'heading': self.heading,
                'total_distance': self.total_distance,
                'timestamp': time.time()
            }
            with open('location_data.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Location Save Error]: {e}")
    
    def load_location(self):
        """Load saved location from file."""
        try:
            if os.path.exists('location_data.json'):
                with open('location_data.json', 'r') as f:
                    data = json.load(f)
                    self.position = data.get('position', {'x': 0, 'y': 0})
                    self.heading = data.get('heading', 0)
                    self.total_distance = data.get('total_distance', 0)
                    print(f"[Location]: Restored previous location")
        except Exception as e:
            print(f"[Location Load Error]: {e}")