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


class AIrisSystem:
    def __init__(self):
        print("=" * 50)
        print("       AIris - AI Assistant Initializing")
        print("=" * 50)
        
        # Initialize core components
        print("\n[1/4] Initializing Audio Manager...")
        self.audio_manager = AudioManager()
        
        print("[2/4] Initializing Navigation Manager...")
        self.navigation_manager = NavigationManager(self.audio_manager)
        
        print("[3/4] Initializing Vision AI...")
        self.vision_ai = VisionAI(self.audio_manager)
        
        print("[4/4] Initializing Voice Handler...")
        self.voice_handler = VoiceCommandHandler(
            self.audio_manager,
            self.navigation_manager,
            self.vision_ai
            # No location tracker
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
        print("   - 'Q' - Quit")
        print("\n" + "=" * 50)
    
    def run(self):
        """Main application loop."""
        print("\n[SYSTEM] Starting main loop...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                break
            
            # Update latest frame for voice handler
            self.voice_handler.latest_frame = frame.copy()
            
            # Navigation mode specific updates
            if self.voice_handler.current_mode == "NAVIGATION":
                # Update navigation (this will return immediately if paused)
                self.navigation_manager.update(time.time())
                
                # Only detect hazards if not paused
                if not self.navigation_manager.is_paused:
                    # Detect hazards
                    detected_objects = self.ui_manager.detect_hazards(frame)
                    
                    # Obstacle avoidance
                    if self.obstacle_avoidance and detected_objects:
                        analysis = self.obstacle_avoidance.analyze_scene(frame)
                        instruction = self.obstacle_avoidance.get_avoidance_instruction(analysis)
                        if instruction:
                            self.audio_manager.speak(instruction, force=False)
            
            # Render UI
            self.ui_manager.render(frame)
            
            # Render navigation status if in navigation mode or paused
            if self.navigation_manager.navigation_started or self.navigation_manager.is_paused:
                self.ui_manager._render_navigation_status(frame, self.navigation_manager)
            
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
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[SYSTEM] Cleaning up...")
        
        # Release camera
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print(f"\n📊 Performance Stats:")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Total frames: {self.frame_count}")
        
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