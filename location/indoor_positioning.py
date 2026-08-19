import time
import math
import threading
from collections import deque


class IndoorPositioning:
    """Indoor positioning using WiFi/Bluetooth signal strength."""
    
    def __init__(self):
        self.wifi_landmarks = {
            'Router_1': {'x': 0, 'y': 0, 'floor': 0, 'signal_strength': -40},
            'Router_2': {'x': 50, 'y': 0, 'floor': 0, 'signal_strength': -45},
            'Router_3': {'x': 0, 'y': 50, 'floor': 0, 'signal_strength': -50},
            'Router_4': {'x': 50, 'y': 50, 'floor': 1, 'signal_strength': -55},
        }
        
        self.signal_history = deque(maxlen=20)
        self.position_estimates = deque(maxlen=10)
        self.current_position = {'x': 0, 'y': 0, 'floor': 0}
        self.confidence = 0
        
        # Try to initialize WiFi scanning
        self.has_wifi = self._init_wifi_scanning()
        
        if self.has_wifi:
            # Start WiFi scanning thread
            threading.Thread(target=self._wifi_scan_loop, daemon=True).start()
    
    def _init_wifi_scanning(self):
        """Initialize WiFi scanning capabilities."""
        try:
            # Try different WiFi libraries
            try:
                import wifi
                self.wifi_module = wifi
                print("[WiFi]: WiFi module loaded")
                return True
            except ImportError:
                pass
            
            try:
                import pywifi
                self.wifi_module = pywifi
                print("[WiFi]: pywifi module loaded")
                return True
            except ImportError:
                pass
            
            print("[WiFi]: No WiFi scanning module available")
            return False
            
        except Exception as e:
            print(f"[WiFi]: Initialization failed: {e}")
            return False
    
    def _wifi_scan_loop(self):
        """Background WiFi scanning loop."""
        while self.has_wifi:
            try:
                # Get current WiFi signals (platform-specific)
                signals = self._scan_wifi()
                
                if signals:
                    self.signal_history.append({
                        'time': time.time(),
                        'signals': signals
                    })
                    
                    # Estimate position
                    position = self.estimate_position(signals)
                    if position:
                        self.position_estimates.append(position)
                        self.current_position = self._smooth_position()
                        self.confidence = self._calculate_confidence()
                
                time.sleep(2)  # Scan every 2 seconds
                
            except Exception as e:
                time.sleep(5)  # Retry in 5 seconds on error
    
    def _scan_wifi(self):
        """Scan for WiFi signals (platform-specific)."""
        # This is a placeholder - actual implementation depends on platform
        # On Linux: use iwlist or nmcli
        # On Windows: use netsh wlan show networks
        # On Android: use Android API
        return {}
    
    def estimate_position(self, signals):
        """Estimate position using trilateration."""
        if len(signals) < 3:
            return None
        
        # Simple weighted centroid method
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for ssid, signal_strength in signals.items():
            if ssid in self.wifi_landmarks:
                landmark = self.wifi_landmarks[ssid]
                
                # Convert signal strength to distance (simplified)
                distance = self._signal_to_distance(signal_strength)
                
                # Weight based on signal strength
                weight = 1.0 / (distance ** 2) if distance > 0 else 1.0
                
                weighted_x += landmark['x'] * weight
                weighted_y += landmark['y'] * weight
                total_weight += weight
        
        if total_weight > 0:
            return {
                'x': weighted_x / total_weight,
                'y': weighted_y / total_weight,
                'floor': self._estimate_floor(signals)
            }
        
        return None
    
    def _signal_to_distance(self, signal_strength):
        """Convert WiFi signal strength to distance (simplified)."""
        # Free space path loss formula (simplified)
        # RSSI = -20 * log10(distance) + RSSI_at_1m
        RSSI_at_1m = -40  # Typical RSSI at 1 meter
        
        if signal_strength >= RSSI_at_1m:
            return 1.0
        
        # Calculate distance using log-distance path loss model
        distance = 10 ** ((RSSI_at_1m - signal_strength) / 20)
        return min(distance, 50)  # Cap at 50 meters
    
    def _estimate_floor(self, signals):
        """Estimate floor based on signal strengths."""
        floor_scores = {}
        
        for ssid, signal_strength in signals.items():
            if ssid in self.wifi_landmarks:
                floor = self.wifi_landmarks[ssid]['floor']
                if floor not in floor_scores:
                    floor_scores[floor] = 0
                floor_scores[floor] += signal_strength
        
        if floor_scores:
            return max(floor_scores, key=floor_scores.get)
        
        return 0
    
    def _smooth_position(self):
        """Smooth position using moving average."""
        if not self.position_estimates:
            return self.current_position
        
        avg_x = sum(p['x'] for p in self.position_estimates) / len(self.position_estimates)
        avg_y = sum(p['y'] for p in self.position_estimates) / len(self.position_estimates)
        avg_floor = sum(p.get('floor', 0) for p in self.position_estimates) / len(self.position_estimates)
        
        return {
            'x': avg_x,
            'y': avg_y,
            'floor': round(avg_floor)
        }
    
    def _calculate_confidence(self):
        """Calculate confidence in position estimate."""
        if len(self.position_estimates) < 3:
            return 0
        
        # Calculate variance of estimates
        xs = [p['x'] for p in self.position_estimates]
        ys = [p['y'] for p in self.position_estimates]
        
        variance_x = sum((x - self.current_position['x'])**2 for x in xs) / len(xs)
        variance_y = sum((y - self.current_position['y'])**2 for y in ys) / len(ys)
        
        total_variance = variance_x + variance_y
        
        # Convert variance to confidence (0-100%)
        confidence = max(0, 100 - (total_variance * 10))
        return min(confidence, 100)
    
    def get_position(self):
        """Get current indoor position."""
        return {
            'position': self.current_position,
            'confidence': self.confidence
        }
    
    def get_location_description(self):
        """Generate location description for indoor positioning."""
        pos = self.current_position
        confidence = self.confidence
        
        if confidence > 70:
            conf_text = "high confidence"
        elif confidence > 40:
            conf_text = "moderate confidence"
        else:
            conf_text = "low confidence"
        
        return (
            f"Indoor position: approximately {pos['x']:.1f} meters east and "
            f"{pos['y']:.1f} meters north, on floor {pos['floor']}. "
            f"Position estimation has {conf_text} ({confidence:.0f}%)."
        )