import cv2
import time
from airis.audio_manager import AudioManager
from airis.navigation import NavigationManager
from airis.vision_ai import VisionAI
from airis.ui_manager import UIManager
from airis.obstacle_avoidance import ObstacleAvoidance
from airis.config import WALKING_SPEED_MPS, ENABLE_OBSTACLE_AVOIDANCE, UI_SHOW_FPS
from airis.conversational_voice_handler import ConversationalVoiceHandler
from airis.location import LocationTracker
from airis.llm_manager import LLMManager


class AIrisSystem:
    def __init__(self):
        print("=" * 50)
        print("       AIris - AI Assistant Initializing")
        print("=" * 50)
        
        print("\n[1/6] Initializing Audio Manager...")
        self.audio_manager = AudioManager()
        
        print("[2/6] Initializing LLM Manager...")
        self.llm_manager = LLMManager(self.audio_manager)
        
        print("[3/6] Initializing Navigation Manager...")
        self.navigation_manager = NavigationManager(self.audio_manager)
        self.location_tracker = LocationTracker()
        self.navigation_manager.set_location_tracker(self.location_tracker)
        
        print("[4/6] Initializing Vision AI...")
        self.vision_ai = VisionAI(self.audio_manager, self.llm_manager)
        
        print("[5/6] Location tracking ready (simulated dead reckoning)")
        
        print("[6/6] Initializing Voice Handler...")
        self.voice_handler = ConversationalVoiceHandler(
            self.audio_manager,
            self.navigation_manager,
            self.vision_ai,
            self.location_tracker,
            self.llm_manager
        )
        
        print("\nInitializing UI and Obstacle Avoidance...")
        self.ui_manager = UIManager(self.audio_manager, self.navigation_manager, self.voice_handler)
        self.obstacle_avoidance = ObstacleAvoidance() if ENABLE_OBSTACLE_AVOIDANCE else None
        
        print("\nInitializing Camera...")
        self.cap = self.initialize_camera()
        
        if self.cap is None:
            print("\n[ERROR]: No camera available!")
            print("Troubleshooting:")
            print("1. Close other apps using the camera (Zoom, Teams, etc.)")
            print("2. Check Windows Settings > Privacy > Camera")
            print("3. Update camera drivers")
            print("4. Try a different USB port")
            print("5. Restart your computer")
            exit(1)
        
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\n" + "=" * 50)
        print("       AIris Active - All Systems Ready")
        print("=" * 50)
        
        print("\nLocation: simulated dead-reckoning from walking simulation")
        
        print("\nLLM Configuration:")
        print(f"   - Text Model: {self.llm_manager.text_model}")
        print(f"   - Vision Model: {self.llm_manager.vision_model}")
        
        print("\nCamera:")
        print("   - Resolution: 640x480")
        print("   - FPS: 30")
        
        print("\nVoice Commands:")
        print("   - 'Go to [place]' - Navigate to destination")
        print("   - 'Cancel Navigation' - Stop navigation completely")
        print("   - 'Pause Navigation' - Pause current route")
        print("   - 'Resume Navigation' - Resume paused route")
        print("   - 'Read' - Read text in view")
        print("   - 'Describe' - Describe scene")
        print("   - 'Mute/Unmute' - Toggle alerts")
        print("   - 'Switch Mode' - Toggle between modes")
        
        print("\nKeyboard Controls:")
        print("   - 'M' - Toggle Mode (Pause/Resume Navigation)")
        print("   - 'C' - Toggle Conversation Mode")
        print("   - 'H' - Help")
        print("   - 'S' - LLM Statistics")
        print("   - 'L' - Location")
        print("   - 'Q' - Quit")
        print("\n" + "=" * 50)
    
    def initialize_camera(self):
        """Initialize camera with fallback options."""
        cv2.destroyAllWindows()
        time.sleep(0.5)
        
        backends = [
            (cv2.CAP_DSHOW, 0),
            (cv2.CAP_MSMF, 0),
            (cv2.CAP_ANY, 0),
            (cv2.CAP_DSHOW, 1),
        ]
        
        for backend, index in backends:
            try:
                print(f"[CAMERA]: Trying backend={backend}, index={index}")
                cap = cv2.VideoCapture(index, backend)
                
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[CAMERA]: Success! Backend={backend}, index={index}")
                        
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"[CAMERA]: Actual settings - {actual_width}x{actual_height} @ {actual_fps}fps")
                        
                        return cap
                    else:
                        print("[CAMERA]: Opened but can't read frame")
                        cap.release()
                else:
                    print("[CAMERA]: Failed to open")
                    
            except Exception as e:
                print(f"[CAMERA]: Error with backend={backend}: {e}")
        
        return None
    
    def run(self):
        """Main application loop."""
        print("\n[SYSTEM] Starting main loop...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING]: Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            self.voice_handler.latest_frame = frame.copy()
            
            if self.voice_handler.current_mode == "NAVIGATION":
                remaining_distance = self.navigation_manager.update(time.time())
                
                if remaining_distance is not None and not self.navigation_manager.is_paused:
                    self.location_tracker.update_position(WALKING_SPEED_MPS * 0.1)
                
                if not self.navigation_manager.is_paused:
                    detected_objects = self.ui_manager.detect_hazards(frame)
                    
                    if self.obstacle_avoidance and detected_objects:
                        analysis = self.obstacle_avoidance.analyze_scene(frame)
                        instruction = self.obstacle_avoidance.get_avoidance_instruction(analysis)
                        if instruction:
                            self.audio_manager.speak(instruction, force=False)
            
            self.ui_manager.render(frame, self.location_tracker)
            
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if UI_SHOW_FPS and elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                self.fps_history.append(fps)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 100, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("AIris Live Field-of-View", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[SYSTEM] Shutting down...")
                break
            elif key in (ord('m'), ord('M')):
                self._toggle_mode()
            elif key in (ord('c'), ord('C')):
                self._toggle_conversation_mode()
            elif key in (ord('h'), ord('H')):
                self.ui_manager.show_help = not self.ui_manager.show_help
                self.ui_manager.help_expiry = time.time() + 30
            elif key in (ord('s'), ord('S')):
                self._show_llm_stats()
            elif key in (ord('l'), ord('L')):
                self._speak_location()
        
        self.cleanup()

    def _toggle_conversation_mode(self):
        """Toggle conversation mode on/off."""
        self.voice_handler.conversation_active = not self.voice_handler.conversation_active
        status = "on" if self.voice_handler.conversation_active else "off"
        self.audio_manager.speak(f"Conversation mode {status}", force=True)
        print(f"[CONVERSATION]: Mode {status}")
    
    def _toggle_mode(self):
        """Toggle between reading and navigation modes with pause/resume."""
        print(f"\n[MODE TOGGLE]: Current mode: {self.voice_handler.current_mode}")
        print(f"[MODE TOGGLE]: Navigation started: {self.navigation_manager.navigation_started}")
        print(f"[MODE TOGGLE]: Navigation paused: {self.navigation_manager.is_paused}")
        print(f"[MODE TOGGLE]: Route completed: {self.navigation_manager.route_completed}")
        
        if self.voice_handler.current_mode == "NAVIGATION":
            self.voice_handler.current_mode = "READING"
            
            if (self.navigation_manager.navigation_started and 
                not self.navigation_manager.route_completed and 
                not self.navigation_manager.is_paused):
                self.navigation_manager.pause_navigation()
                self.audio_manager.speak("Switched to reading mode. Navigation paused.", force=True)
                print("[MODE]: Switched to READING mode (navigation paused)")
            elif self.navigation_manager.is_paused:
                self.audio_manager.speak("Switched to reading mode. Navigation is paused.", force=True)
                print("[MODE]: Switched to READING mode (already paused)")
            else:
                self.audio_manager.speak("Switched to reading mode.", force=True)
                print("[MODE]: Switched to READING mode")
        else:
            self.voice_handler.current_mode = "NAVIGATION"
            
            if self.navigation_manager.is_paused:
                self.navigation_manager.resume_navigation()
                print("[MODE]: Switched to NAVIGATION mode (navigation resumed)")
            elif (self.navigation_manager.navigation_started and 
                  not self.navigation_manager.route_completed):
                self.audio_manager.speak("Switched to navigation mode. Continuing route.", force=True)
                print("[MODE]: Switched to NAVIGATION mode (continuing route)")
            else:
                self.audio_manager.speak("Switched to navigation mode.", force=True)
                print("[MODE]: Switched to NAVIGATION mode")
    
    def _show_llm_stats(self):
        """Show LLM statistics."""
        stats = self.llm_manager.get_stats()
        print("\n" + "=" * 50)
        print("           LLM Statistics")
        print("=" * 50)
        print(f"Text Model: {stats['text_model']}")
        print(f"Vision Model: {stats['vision_model']}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"API Errors: {stats['api_errors']}")
        print(f"Cache Size: {stats['cache_size']}")
        print(f"Requests/Min: {stats['requests_last_minute']}")
        print("Model Usage:")
        for model, count in stats['model_usage'].items():
            print(f"  - {model}: {count} requests")
        print("=" * 50)
        
        summary = f"Total requests: {stats['total_requests']}, Cache hits: {stats['cache_hits']}"
        self.audio_manager.speak(summary, force=True)
    
    def _speak_location(self):
        """Speak current location."""
        description = self.location_tracker.get_location_description()
        self.audio_manager.speak(description, force=True)
        
        nearest = self.location_tracker.get_nearest_landmark()
        if nearest:
            landmark_info = f"Nearest landmark: {nearest['name']} ({nearest['distance']:.1f}m)"
            self.audio_manager.speak(landmark_info, force=False)
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[SYSTEM] Cleaning up...")
        
        self.location_tracker.save_location()
        self.llm_manager.save_cache()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            print("\nPerformance Stats:")
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Total frames: {self.frame_count}")
        
        stats = self.llm_manager.get_stats()
        print("\nLLM Stats:")
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
