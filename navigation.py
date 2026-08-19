import time
from config import WALKING_SPEED_MPS


class NavigationManager:
    def __init__(self, audio_manager, voice_handler=None):
        self.audio_manager = audio_manager
        self.voice_handler = voice_handler
        self.active_route = []
        self.current_step_index = 0
        self.step_start_time = 0
        self.announced_preturn = False
        self.announced_turn = False
        self.current_destination = None
        
        # Navigation state
        self.is_paused = False
        self.pause_time = 0
        self.total_pause_duration = 0
        self.navigation_started = False
        self.route_completed = False
        
        # Store remaining distance for UI
        self.remaining_distance = 0
        self.current_step_progress = 0  # 0 to 1
    
    def set_voice_handler(self, voice_handler):
        """Set reference to voice handler for mode switching."""
        self.voice_handler = voice_handler
    
    def start_route(self, destination_item):
        """Initializes time-based route simulation."""
        self.active_route = destination_item.get("route", [])
        self.current_step_index = 0
        self.step_start_time = time.time()
        self.announced_preturn = False
        self.announced_turn = False
        self.current_destination = destination_item
        self.is_paused = False
        self.pause_time = 0
        self.total_pause_duration = 0
        self.navigation_started = True
        self.route_completed = False
        self.remaining_distance = 0
        self.current_step_progress = 0

        if self.active_route:
            first_step = self.active_route[0]
            self.remaining_distance = first_step["distance_meters"]
            self.current_step_progress = 0
            self.audio_manager.speak(
                f"Starting route to {destination_item['name']}. {first_step['instruction']}.", 
                force=True
            )
    
    def pause_navigation(self):
        """Pause navigation without canceling."""
        if self.navigation_started and not self.route_completed and self.active_route and not self.is_paused:
            self.is_paused = True
            self.pause_time = time.time()
            print(f"[NAVIGATION]: Navigation paused at {self.remaining_distance:.1f}m remaining")
    
    def resume_navigation(self):
        """Resume paused navigation."""
        if self.is_paused and self.active_route:
            # Adjust step_start_time to account for pause duration
            pause_duration = time.time() - self.pause_time
            self.total_pause_duration += pause_duration
            self.step_start_time += pause_duration
            self.is_paused = False
            self.pause_time = 0
            
            # Announce resume
            if self.current_step_index < len(self.active_route):
                step = self.active_route[self.current_step_index]
                self.audio_manager.speak(
                    f"Resuming navigation. {step['instruction']}.", 
                    force=True
                )
                print(f"[NAVIGATION]: Navigation resumed at {self.remaining_distance:.1f}m remaining")
    
    def toggle_pause(self):
        """Toggle between pause and resume."""
        if self.is_paused:
            self.resume_navigation()
        else:
            self.pause_navigation()
    
    def cancel_navigation(self):
        """Cancels active navigation route and returns to reading mode."""
        self.active_route = []
        self.current_step_index = 0
        self.current_destination = None
        self.is_paused = False
        self.pause_time = 0
        self.total_pause_duration = 0
        self.navigation_started = False
        self.route_completed = False
        self.remaining_distance = 0
        self.current_step_progress = 0
        
        self.audio_manager.speak("Navigation cancelled. Returning to reading mode.", force=True)
        
        # Return to reading mode if voice handler is available
        if self.voice_handler:
            self.voice_handler.current_mode = "READING"
            print("[MODE]: Returned to READING mode")
    
    def update(self, current_time):
        """Updates navigation state based on elapsed time."""
        # Don't update if paused
        if self.is_paused:
            return self.remaining_distance
        
        if not self.active_route or self.current_step_index >= len(self.active_route):
            return None
        
        step = self.active_route[self.current_step_index]
        total_dist = step["distance_meters"]
        turn_action = step["turn"]

        # Calculate distance covered based on elapsed walking time
        # Subtract total pause duration to account for paused time
        elapsed_time = current_time - self.step_start_time
        distance_covered = elapsed_time * WALKING_SPEED_MPS
        self.remaining_distance = max(0.0, total_dist - distance_covered)
        
        # Update progress for UI
        self.current_step_progress = min(1.0, distance_covered / total_dist if total_dist > 0 else 0)

        # Trigger pre-turn alert
        if self.remaining_distance <= 10.0 and self.remaining_distance > 3.0 and not self.announced_preturn:
            self.audio_manager.speak(f"In 10 meters, {turn_action.lower()}.", force=True)
            self.announced_preturn = True

        # Trigger turn command
        if self.remaining_distance <= 3.0 and not self.announced_turn:
            if "arrived" in turn_action.lower():
                self.audio_manager.speak(turn_action, force=True)
            else:
                self.audio_manager.speak(f"Please {turn_action.lower()} now.", force=True)
            self.announced_turn = True

        # Advance to next step
        if self.remaining_distance <= 0.0:
            self.current_step_index += 1
            self.step_start_time = time.time()
            self.announced_preturn = False
            self.announced_turn = False
            self.current_step_progress = 0

            if self.current_step_index < len(self.active_route):
                next_step = self.active_route[self.current_step_index]
                self.remaining_distance = next_step["distance_meters"]
                self.audio_manager.speak(next_step["instruction"], force=True)
            else:
                # Route completed
                self.route_completed = True
                self.navigation_started = False
                self.remaining_distance = 0
                self.audio_manager.speak("Route completed. Returning to reading mode.", force=True)
                self.active_route = []
                self.current_destination = None
                
                # Return to reading mode if voice handler is available
                if self.voice_handler:
                    self.voice_handler.current_mode = "READING"
                    print("[MODE]: Returned to READING mode")
        
        return self.remaining_distance
    
    def get_remaining_distance(self):
        """Get current remaining distance for UI display."""
        return self.remaining_distance
    
    def get_current_step_progress(self):
        """Get current step progress (0-1) for UI progress bar."""
        return self.current_step_progress
    
    def get_status(self):
        """Get current navigation status."""
        return {
            'active': self.navigation_started,
            'paused': self.is_paused,
            'completed': self.route_completed,
            'current_step': self.current_step_index,
            'total_steps': len(self.active_route),
            'destination': self.current_destination.get('name') if self.current_destination else None,
            'remaining_distance': self.remaining_distance,
            'progress': self.current_step_progress
        }