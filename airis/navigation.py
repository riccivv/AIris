import time
import math
from .config import WALKING_SPEED_MPS


class NavigationManager:
    def __init__(self, audio_manager, voice_handler=None, location_tracker=None):
        self.audio_manager = audio_manager
        self.voice_handler = voice_handler
        self.location_tracker = location_tracker
        
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
        self.current_step_progress = 0
    
    def set_voice_handler(self, voice_handler):
        """Set reference to voice handler for mode switching."""
        self.voice_handler = voice_handler
    
    def set_location_tracker(self, location_tracker):
        """Set reference to location tracker."""
        self.location_tracker = location_tracker
    
    def check_destination_validity(self, destination_item):
        """Check if destination is valid (not already there)."""
        if not self.location_tracker:
            return True, None
        
        # Find landmark in location tracker
        landmark = self.location_tracker.find_landmark(destination_item['name'])
        
        if landmark:
            # Check if user is already at this location
            if self.location_tracker.is_at_location(destination_item['name']):
                return False, "already_there"
            
            # Check if destination is reasonable (not too far)
            distance = self.location_tracker.get_distance_to(landmark)
            if distance > 1000:  # More than 1km
                return False, "too_far"
        
        return True, None
    
    def start_route(self, destination_item):
        """Start navigation with validation."""
        # Check if destination is valid
        is_valid, reason = self.check_destination_validity(destination_item)
        
        if not is_valid:
            if reason == "already_there":
                self.audio_manager.speak(
                    f"You are already at {destination_item['name']}. No navigation needed.", 
                    force=True
                )
            elif reason == "too_far":
                self.audio_manager.speak(
                    f"{destination_item['name']} is quite far. Consider alternative transportation.", 
                    force=True
                )
            return False
        
        # Set current destination in location tracker
        if self.location_tracker:
            self.location_tracker.current_destination = destination_item
            self.location_tracker.arrived_at_destination = False
        
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
            self.audio_manager.speak(
                f"Starting route to {destination_item['name']}. {first_step['instruction']}.", 
                force=True
            )
        else:
            # Create default route if none exists
            self._create_default_route(destination_item)
        
        return True
    
    def _create_default_route(self, destination_item):
        """Create a default route based on distance to destination."""
        if not self.location_tracker:
            return
        
        direction_info = self.location_tracker.get_direction_to(destination_item)
        
        self.active_route = [
            {
                "instruction": f"Head {direction_info['direction']} for {int(direction_info['distance'])} meters",
                "distance_meters": direction_info['distance'],
                "turn": f"You have arrived at {destination_item['name']}"
            }
        ]
        
        self.current_step_index = 0
        self.step_start_time = time.time()
        self.remaining_distance = direction_info['distance']
        
        self.audio_manager.speak(
            f"Head {direction_info['direction']} for {int(direction_info['distance'])} meters to reach {destination_item['name']}.", 
            force=True
        )
    
    def pause_navigation(self):
        """Pause current navigation."""
        if not self.navigation_started or self.route_completed:
            return
        self.is_paused = True
        self.pause_time = time.time()
    
    def resume_navigation(self):
        """Resume a paused navigation, excluding the pause duration from elapsed time."""
        if not self.is_paused:
            return
        pause_duration = time.time() - self.pause_time
        self.total_pause_duration += pause_duration
        self.step_start_time += pause_duration
        self.is_paused = False
    
    def cancel_navigation(self):
        """Cancel the current navigation and reset all navigation state."""
        self.active_route = []
        self.current_step_index = 0
        self.current_destination = None
        self.navigation_started = False
        self.is_paused = False
        self.route_completed = False
        self.remaining_distance = 0
        self.current_step_progress = 0
        self.announced_preturn = False
        self.announced_turn = False
        if self.location_tracker:
            self.location_tracker.current_destination = None
            self.location_tracker.arrived_at_destination = False
    
    def update(self, current_time):
        """Updates navigation state based on elapsed time."""
        if self.is_paused:
            return self.remaining_distance
        
        if not self.active_route or self.current_step_index >= len(self.active_route):
            return None
        
        step = self.active_route[self.current_step_index]
        total_dist = step["distance_meters"]
        turn_action = step["turn"]

        # Calculate distance covered
        elapsed_time = current_time - self.step_start_time
        distance_covered = elapsed_time * WALKING_SPEED_MPS
        self.remaining_distance = max(0.0, total_dist - distance_covered)
        
        # Update progress
        self.current_step_progress = min(1.0, distance_covered / total_dist if total_dist > 0 else 0)
        
        # Update location tracker based on movement
        if self.location_tracker and distance_covered > 0:
            # Assume user is moving toward destination
            if self.current_destination:
                direction_info = self.location_tracker.get_direction_to(self.current_destination)
                heading = direction_info['angle']
                self.location_tracker.update_position(
                    WALKING_SPEED_MPS * 0.1,  # Small increment
                    heading=heading
                )

        # Trigger alerts (only for actual turn steps - straight steps use the instruction)
        is_turn_step = "left" in turn_action.lower() or "right" in turn_action.lower()
        if is_turn_step:
            if self.remaining_distance <= 10.0 and self.remaining_distance > 3.0 and not self.announced_preturn:
                self.audio_manager.speak(f"In 10 meters, {turn_action.lower()}.", force=True)
                self.announced_preturn = True

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
                # Speak the instruction only for straight/continue steps.
                # Turn steps are announced by the 10m + turn-now alerts instead.
                next_turn = next_step.get("turn", "").lower()
                if "left" not in next_turn and "right" not in next_turn:
                    self.audio_manager.speak(next_step["instruction"], force=True)
            else:
                # Route completed
                self.route_completed = True
                self.navigation_started = False
                self.remaining_distance = 0
                
                # Update location tracker
                if self.location_tracker:
                    self.location_tracker.arrived_at_destination = True
                    self.location_tracker.current_destination = None
                
                self.audio_manager.speak("Route completed. Returning to reading mode.", force=True)
                self.active_route = []
                self.current_destination = None
                
                if self.voice_handler:
                    self.voice_handler.current_mode = "READING"
        
        return self.remaining_distance
    
    def get_remaining_distance(self):
        return self.remaining_distance
    
    def get_current_step_progress(self):
        return self.current_step_progress
    
    def get_status(self):
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