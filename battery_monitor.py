import psutil
import threading
import time


class BatteryMonitor:
    """Monitor system battery status for mobile deployments."""
    
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
        self.low_battery_threshold = 20
        self.critical_battery_threshold = 10
        self.last_warning_time = 0
        self.warning_cooldown = 300  # 5 minutes
        
        # Start monitoring thread
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """Background battery monitoring loop."""
        while True:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    percent = battery.percent
                    plugged = battery.power_plugged
                    
                    current_time = time.time()
                    
                    if (not plugged and 
                        percent <= self.low_battery_threshold and 
                        current_time - self.last_warning_time > self.warning_cooldown):
                        
                        if percent <= self.critical_battery_threshold:
                            self.audio_manager.speak(
                                f"Warning: Battery critically low at {percent}%. Please charge immediately.", 
                                force=True
                            )
                        else:
                            self.audio_manager.speak(
                                f"Battery at {percent}%. Consider charging soon.", 
                                force=True
                            )
                        
                        self.last_warning_time = current_time
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"[Battery Monitor Error]: {e}")
                time.sleep(300)  # Retry in 5 minutes on error