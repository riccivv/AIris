import time
import math
import threading
from .base_tracker import LocationTracker


class GPSLocationTracker(LocationTracker):
    """Enhanced location tracking with GPS support."""
    
    def __init__(self):
        super().__init__()
        self.has_gps = False
        self.gps_data = None
        self.gps_thread = None
        
        # Try to initialize GPS
        self._init_gps()
    
    def _init_gps(self):
        """Initialize GPS if available."""
        try:
            import gps
            self.gps_session = gps.gps(mode=gps.WATCH_ENABLE)
            self.has_gps = True
            print("[GPS]: GPS initialized successfully")
            
            # Start GPS update thread
            self.gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
            self.gps_thread.start()
        except ImportError:
            print("[GPS]: GPS module not available")
        except Exception as e:
            print(f"[GPS]: Initialization failed: {e}")
    
    def _gps_loop(self):
        """Background GPS data collection loop."""
        while self.has_gps:
            try:
                report = self.gps_session.next()
                if report['class'] == 'TPV':
                    self.gps_data = {
                        'lat': getattr(report, 'lat', 0),
                        'lon': getattr(report, 'lon', 0),
                        'alt': getattr(report, 'alt', 0),
                        'speed': getattr(report, 'speed', 0),
                        'heading': getattr(report, 'track', 0),
                        'time': getattr(report, 'time', time.time())
                    }
                    
                    # Update position from GPS
                    self.update_from_gps(
                        self.gps_data['lat'],
                        self.gps_data['lon'],
                        self.gps_data['alt']
                    )
                    
                    # Update heading if GPS provides it
                    if self.gps_data['heading'] > 0:
                        self.heading = self.gps_data['heading']
            except Exception as e:
                time.sleep(1)  # Wait before retry
    
    def get_gps_coordinates(self):
        """Get raw GPS coordinates."""
        return self.gps_data
    
    def get_speed(self):
        """Get current speed from GPS."""
        if self.gps_data and 'speed' in self.gps_data:
            return self.gps_data['speed']
        return 0
    
    def get_location_description(self):
        """Override to include GPS info."""
        base_description = super().get_location_description()
        
        if self.has_gps and self.gps_data:
            coords = f"GPS coordinates: {self.gps_data['lat']:.6f}, {self.gps_data['lon']:.6f}"
            return f"{base_description} {coords}."
        
        return base_description