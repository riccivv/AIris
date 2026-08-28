import cv2
import numpy as np
from collections import deque
import time


class ObstacleAvoidance:
    """Advanced obstacle detection and avoidance system."""
    
    def __init__(self):
        self.obstacle_history = deque(maxlen=30)
        self.last_avoidance_time = 0
        self.avoidance_cooldown = 3.0
        self.safe_zones = []
        
    def analyze_scene(self, frame, depth_map=None):
        """Analyze scene for obstacles and suggest avoidance paths."""
        height, width = frame.shape[:2]
        
        # Divide frame into zones
        zones = {
            'left': frame[:, :width//3],
            'center': frame[:, width//3:2*width//3],
            'right': frame[:, 2*width//3:]
        }
        
        obstacles = {}
        for zone_name, zone_frame in zones.items():
            # Simple obstacle detection (can be enhanced with depth sensing)
            gray = cv2.cvtColor(zone_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            obstacle_density = np.mean(edges)
            obstacles[zone_name] = obstacle_density
        
        # Find clearest path
        clearest_zone = min(obstacles, key=obstacles.get)
        
        return {
            'obstacles': obstacles,
            'clearest_path': clearest_zone,
            'should_avoid': any(density > 50 for density in obstacles.values())
        }
    
    def get_avoidance_instruction(self, analysis):
        """Generate avoidance instructions."""
        if not analysis['should_avoid']:
            return None
        
        current_time = time.time()
        if current_time - self.last_avoidance_time < self.avoidance_cooldown:
            return None
        
        self.last_avoidance_time = current_time
        
        clearest = analysis['clearest_path']
        if clearest == 'left':
            return "Obstacle ahead. Move to the left."
        elif clearest == 'right':
            return "Obstacle ahead. Move to the right."
        else:
            return "Proceed with caution."