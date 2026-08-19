import cv2
import time
import threading
from audio_manager import AudioManager
from navigation import NavigationManager
from vision_ai import VisionAI
from voice_recognition import VoiceCommandHandler
from ui_manager import UIManager
from obstacle_avoidance import ObstacleAvoidance
from config import *

# Location imports
from location import LocationTracker, GPSLocationTracker, IndoorPositioning, MapIntegration

# LLM Manager import
from llm_manager import LLMManager


class AIrisSystem:
    def __init__(self):
        print("=" * 50)
        print("       AIris - AI Assistant Initializing")
        print("=" * 50)
        
        # Initialize core components in correct order
        print("\n[1/6] Initializing Audio Manager...")
        self.audio_manager = AudioManager()  # Create audio manager FIRST
        
        print("[2/6] Initializing LLM Manager...")
        self.llm_manager = LLMManager(self.audio_manager)  # Now audio_manager exists
        
        print("[3/6] Initializing Navigation Manager...")
        self.navigation_manager = NavigationManager(self.audio_manager)
        
        print("[4/6] Initializing Vision AI...")
        self.vision_ai = VisionAI(self.audio_manager)
        self.vision_ai.llm_manager = self.llm_manager  # Inject LLM manager
        
        print("[5/6] Initializing Location Services...")
        # Initialize location tracking
        self.location_tracker = LocationTracker()
        self.indoor_positioning = IndoorPositioning()
        self.map_integration = MapIntegration()
        self.gps_tracker = GPSLocationTracker()
        
        print("[6/6] Initializing Voice Handler...")
        self.voice_handler = VoiceCommandHandler(
            self.audio_manager,
            self.navigation_manager,
            self.vision_ai,
            self.location_tracker
        )
        
        # Initialize UI and obstacle avoidance
        print("\nInitializing UI and Obstacle Avoidance...")
        self.ui_manager = UIManager(self.audio_manager, self.navigation_manager, self.voice_handler)
        self.obstacle_avoidance = ObstacleAvoidance()
        
        # Initialize video capture
        print("Initializing Camera...")
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Print status
        print("\n" + "=" * 50)
        print("       AIris Active - All Systems Ready")
        print("=" * 50)
        print("\n📍 Location Services:")
        print(f"   - GPS: {'Available' if self.gps_tracker.has_gps else 'Not available (using dead reckoning)'}")
        print(f"   - Indoor Positioning: {'Available' if self.indoor_positioning.has_wifi else 'Not available'}")
        print(f"   - Map Provider: {self.map_integration.map_provider}")
        
        print("\n🤖 LLM Configuration:")
        print(f"   - Primary Model: {self.llm_manager.primary_model}")
        print(f"   - Fallback Model: {self.llm_manager.fallback_model}")
        
        print("\n🎤 Voice Commands:")
        print("   - 'Go to [place]' - Navigate to destination")
        print("   - 'Cancel Navigation' - Stop navigation completely")
        print("   - 'Pause Navigation' - Pause current route")
        print("   - 'Resume Navigation' - Resume paused route")
        print("   - 'Read' - Read text in view")
        print("   - 'Describe' - Describe scene")
        print("   - 'Mute/Unmute' - Toggle alerts")
        print("   - 'Switch Mode' - Toggle between modes")
        
        print("\n⌨️  Keyboard Controls:")
        print("   - 'M' - Toggle Mode (Pause/Resume Navigation)")
        print("   - 'H' - Help")
        print("   - 'S' - LLM Statistics")
        print("   - 'Q' - Quit")
        print("\n" + "=" * 50)
    
    def run(self):
        """Main application loop."""
        print("\n[SYSTEM] Starting main loop...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING]: Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Update latest frame for voice handler
            self.voice_handler.latest_frame = frame.copy()
            
            # Update location from indoor positioning
            if self.indoor_positioning and self.indoor_positioning.has_wifi:
                indoor_pos = self.indoor_positioning.get_position()
                if indoor_pos['confidence'] > 60:
                    self.location_tracker.position['x'] = indoor_pos['position']['x']
                    self.location_tracker.position['y'] = indoor_pos['position']['y']
                    self.location_tracker.floor = indoor_pos['position']['floor']
            
            # Navigation mode specific updates
            if self.voice_handler.current_mode == "NAVIGATION":
                # Update navigation
                remaining_distance = self.navigation_manager.update(time.time())
                
                # Update location tracker based on movement
                if remaining_distance is not None and not self.navigation_manager.is_paused:
                    self.location_tracker.update_position(WALKING_SPEED_MPS * 0.1)
                
                # Only detect hazards if not paused
                if not self.navigation_manager.is_paused:
                    detected_objects = self.ui_manager.detect_hazards(frame)
                    
                    # Obstacle avoidance
                    if self.obstacle_avoidance and detected_objects:
                        analysis = self.obstacle_avoidance.analyze_scene(frame)
                        instruction = self.obstacle_avoidance.get_avoidance_instruction(analysis)
                        if instruction:
                            self.audio_manager.speak(instruction, force=False)
            
            # Render UI (with or without location tracker)
            if self.location_tracker:
                self.ui_manager.render(frame, self.location_tracker)
            else:
                self.ui_manager.render(frame)
            
            # Calculate and display FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                self.fps_history.append(fps)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 100, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("AIris Live Field-of-View", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] Shutting down...")
                break
            elif key in (ord('m'), ord('M')):
                self._toggle_mode()
            elif key in (ord('h'), ord('H')):
                self.ui_manager.show_help = not self.ui_manager.show_help
                self.ui_manager.help_expiry = time.time() + 30
            elif key in (ord('s'), ord('S')):
                self._show_llm_stats()
            elif key in (ord('l'), ord('L')):
                self._speak_location()
        
        # Cleanup
        self.cleanup()
    
    def _toggle_mode(self):
        """Toggle between reading and navigation modes with pause/resume."""
        print(f"\n[MODE TOGGLE]: Current mode: {self.voice_handler.current_mode}")
        print(f"[MODE TOGGLE]: Navigation started: {self.navigation_manager.navigation_started}")
        print(f"[MODE TOGGLE]: Navigation paused: {self.navigation_manager.is_paused}")
        print(f"[MODE TOGGLE]: Route completed: {self.navigation_manager.route_completed}")
        
        if self.voice_handler.current_mode == "NAVIGATION":
            # Switch to reading mode (pause navigation if active)
            self.voice_handler.current_mode = "READING"
            
            # Check if there's active navigation
            if (self.navigation_manager.navigation_started and 
                not self.navigation_manager.route_completed and 
                not self.navigation_manager.is_paused):
                # Pause navigation instead of canceling
                self.navigation_manager.pause_navigation()
                self.audio_manager.speak("Switched to reading mode. Navigation paused.", force=True)
                print("[MODE]: Switched to READING mode (navigation paused)")
            elif self.navigation_manager.is_paused:
                # Already paused
                self.audio_manager.speak("Switched to reading mode. Navigation is paused.", force=True)
                print("[MODE]: Switched to READING mode (already paused)")
            else:
                # No active navigation
                self.audio_manager.speak("Switched to reading mode.", force=True)
                print("[MODE]: Switched to READING mode")
        else:
            # Switch to navigation mode (resume if paused)
            self.voice_handler.current_mode = "NAVIGATION"
            
            if self.navigation_manager.is_paused:
                # Resume paused navigation
                self.navigation_manager.resume_navigation()
                print("[MODE]: Switched to NAVIGATION mode (navigation resumed)")
            elif (self.navigation_manager.navigation_started and 
                  not self.navigation_manager.route_completed):
                # Continue active navigation
                self.audio_manager.speak("Switched to navigation mode. Continuing route.", force=True)
                print("[MODE]: Switched to NAVIGATION mode (continuing route)")
            else:
                # No navigation to resume
                self.audio_manager.speak("Switched to navigation mode.", force=True)
                print("[MODE]: Switched to NAVIGATION mode")
    
    def _show_llm_stats(self):
        """Show LLM statistics."""
        stats = self.llm_manager.get_stats()
        print("\n" + "=" * 50)
        print("           LLM Statistics")
        print("=" * 50)
        print(f"Primary Model: {stats['primary_model']}")
        print(f"Primary Available: {stats['primary_available']}")
        print(f"Fallback Model: {stats['fallback_model']}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"API Errors: {stats['api_errors']}")
        print(f"Cache Size: {stats['cache_size']}")
        print(f"Requests/Min: {stats['requests_last_minute']}")
        print(f"Model Usage: {stats['model_usage']}")
        print("=" * 50)
        
        # Also speak summary
        summary = f"Total requests: {stats['total_requests']}, Cache hits: {stats['cache_hits']}"
        self.audio_manager.speak(summary, force=True)
    
    def _speak_location(self):
        """Speak current location."""
        description = self.location_tracker.get_location_description()
        self.audio_manager.speak(description, force=True)
        
        # Also show nearest landmark if available
        nearest = self.location_tracker.get_nearest_landmark()
        if nearest:
            landmark_info = f"Nearest landmark: {nearest['name']} ({nearest['distance']:.1f}m)"
            self.audio_manager.speak(landmark_info, force=False)
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[SYSTEM] Cleaning up...")
        
        # Save location data
        self.location_tracker.save_location()
        
        # Save LLM cache
        self.llm_manager.save_cache()
        
        # Release camera
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"\n📊 Performance Stats:")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Total frames: {self.frame_count}")
            print(f"   - Total distance traveled: {self.location_tracker.total_distance:.1f}m")
        
        # Print LLM stats
        stats = self.llm_manager.get_stats()
        print(f"\n🤖 LLM Stats:")
        print(f"   - Total requests: {stats['total_requests']}")
        print(f"   - Cache hits: {stats['cache_hits']}")
        print(f"   - API errors: {stats['api_errors']}")
        
        print("[SYSTEM] Shutdown complete.")


def main():
    try:
        system = AIrisSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()