import speech_recognition as sr
import threading
import time


class VoiceCommandHandler:
    def __init__(self, audio_manager, navigation_manager, vision_ai, location_tracker=None):
        self.audio_manager = audio_manager
        self.navigation_manager = navigation_manager
        self.vision_ai = vision_ai
        self.location_tracker = location_tracker
        
        self.current_mode = "READING"
        self.pending_destination = None
        self.latest_frame = None
        
        # Set voice handler reference in navigation manager
        self.navigation_manager.set_voice_handler(self)
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        
        # Start listener thread
        threading.Thread(target=self._listen_loop, daemon=True).start()
    
    def _listen_loop(self):
        """Background microphone worker thread."""
        try:
            microphone = sr.Microphone()
            print("[MIC]: Listener Running Successfully!")
        except Exception as e:
            print(f"[MIC Error]: Listener setup failed: {e}")
            return
        
        while True:
            if self.audio_manager.is_speaking:
                time.sleep(0.2)
                continue
            
            try:
                with microphone as source:
                    audio = self.recognizer.listen(source, phrase_time_limit=5, timeout=2)
                    
                if self.audio_manager.is_speaking:
                    continue
                
                command = self.recognizer.recognize_google(audio).lower()
                print(f"[HEARD]: '{command}'")
                
                # Process command
                self._process_command(command)
                
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                pass
            except Exception as e:
                print(f"[Voice Error]: {e}")
                time.sleep(0.5)
    
    def _process_command(self, command):
        """Process voice commands with full functionality."""
        user_lower = command.lower().strip()
        
        # CANCEL NAVIGATION (highest priority - this actually cancels)
        if any(word in user_lower for word in ["cancel navigation", "stop navigation", "cancel route"]):
            print("[NAVIGATION]: Cancelling navigation")
            self.navigation_manager.cancel_navigation()
            self.pending_destination = None
            return
        
        # PAUSE NAVIGATION
        if any(word in user_lower for word in ["pause navigation", "pause route", "hold navigation"]):
            print("[NAVIGATION]: Pausing navigation")
            if self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
                self.navigation_manager.pause_navigation()
                self.audio_manager.speak("Navigation paused.", force=True)
            else:
                self.audio_manager.speak("No active navigation to pause.", force=True)
            return
        
        # RESUME NAVIGATION
        if any(word in user_lower for word in ["resume navigation", "resume route", "continue navigation", "continue route"]):
            print("[NAVIGATION]: Resuming navigation")
            if self.navigation_manager.is_paused:
                self.navigation_manager.resume_navigation()
            else:
                self.audio_manager.speak("Navigation is not paused.", force=True)
            return
        
        # CONFIRMATION FLOW (if there's a pending destination)
        if self.pending_destination:
            if any(word in user_lower for word in ["yes", "yeah", "confirm", "correct", "sure", "ok", "okay"]):
                print(f"[NAVIGATION]: Confirming destination: {self.pending_destination['name']}")
                destination = self.pending_destination
                self.pending_destination = None
                self.current_mode = "NAVIGATION"
                self.navigation_manager.start_route(destination)
                return
            elif any(word in user_lower for word in ["no", "nope", "stop", "wrong", "cancel"]):
                print("[NAVIGATION]: Destination cancelled")
                self.audio_manager.speak("Destination cancelled. Remaining in reading mode.", force=True)
                self.pending_destination = None
                return
        
        # DESTINATION NAVIGATION REQUEST
        if any(trigger in user_lower for trigger in ["go to", "navigate to", "take me to"]):
            from knowledge_base import rag_lookup_destination
            
            # Extract destination
            target_raw = user_lower
            for trigger in ["go to", "navigate to", "take me to"]:
                if trigger in user_lower:
                    target_raw = user_lower.split(trigger)[-1].strip()
                    break
            
            print(f"[NAVIGATION]: Requesting navigation to: {target_raw}")
            
            matched_item = rag_lookup_destination(target_raw)
            
            if matched_item:
                self.pending_destination = matched_item
                self.audio_manager.speak(
                    f"Found {matched_item['name']}. Would you like to navigate there?", 
                    force=True
                )
            else:
                self.pending_destination = {
                    "name": target_raw.title(),
                    "details": "Custom Location",
                    "route": [{"instruction": f"Head towards {target_raw.title()}", 
                               "distance_meters": 100, "turn": "You have arrived"}]
                }
                self.audio_manager.speak(
                    f"I don't have {target_raw} in my maps. Set as custom destination?", 
                    force=True
                )
            return
        
        # LOCATION QUERIES
        if any(phrase in user_lower for phrase in ["where am i", "current location", "my location", "where are we"]):
            print("[LOCATION]: Location query")
            if self.location_tracker:
                description = self.location_tracker.get_location_description()
                self.audio_manager.speak(description, force=True)
            else:
                self.audio_manager.speak("Location tracking is not available.", force=True)
            return
        
        if any(phrase in user_lower for phrase in ["what's near", "what is near", "nearby", "what's around", "what is around"]):
            print("[LOCATION]: Nearby landmarks query")
            if self.location_tracker:
                nearest = self.location_tracker.get_nearest_landmark()
                if nearest:
                    self.audio_manager.speak(
                        f"The nearest landmark is {nearest['name']}, about {nearest['distance']:.1f} meters away. {nearest.get('description', '')}", 
                        force=True
                    )
                else:
                    self.audio_manager.speak("No known landmarks nearby.", force=True)
            else:
                self.audio_manager.speak("Location tracking is not available.", force=True)
            return
        
        # MODE SWITCHING (with navigation pause/resume)
        if "reading mode" in user_lower:
            print("[MODE]: Switching to reading mode")
            self._switch_to_reading_mode()
            return
        elif "navigation mode" in user_lower or user_lower == "navigate":
            print("[MODE]: Switching to navigation mode")
            self._switch_to_navigation_mode()
            return
        elif "switch mode" in user_lower or "toggle mode" in user_lower or "change mode" in user_lower:
            print("[MODE]: Toggling mode")
            self._toggle_mode()
            return
        
        # MUTE/UNMUTE ALERTS
        if any(word in user_lower for word in ["unmute", "enable alerts", "turn on alerts"]):
            print("[ALERTS]: Unmuting alerts")
            self.audio_manager.set_mute_alerts(False)
            self.audio_manager.speak("Hazard alerts unmuted.", force=True)
            return
        elif any(word in user_lower for word in ["mute", "silence alerts", "disable alerts", "turn off alerts"]):
            print("[ALERTS]: Muting alerts")
            self.audio_manager.set_mute_alerts(True)
            self.audio_manager.speak("Hazard alerts muted.", force=True)
            return
        
        # VISION AI COMMANDS
        if user_lower in ["read", "read this", "read text", "read it", "read the text"]:
            print("[VISION]: Reading text")
            self.audio_manager.speak("Reading text.", force=True)
            self.vision_ai.ask_multimodal(
                "Extract and read all clear visible text in this image out loud.", 
                self.latest_frame
            )
            return
        elif any(phrase in user_lower for phrase in ["describe", "what do you see", "what's in front", "what is in front"]):
            print("[VISION]: Describing scene")
            self.audio_manager.speak("Analyzing scene.", force=True)
            self.vision_ai.ask_multimodal(
                "Describe what is directly in front of the user concisely.", 
                self.latest_frame
            )
            return
        
        # HELP COMMAND
        if "help" in user_lower or "what can i say" in user_lower or "commands" in user_lower:
            print("[HELP]: Providing help")
            self._provide_help()
            return
        
        # If we get here, it wasn't a specific command
        print(f"[INFO]: Unrecognized command: '{command}'")
    
    def _switch_to_reading_mode(self):
        """Switch to reading mode, pausing navigation if active."""
        if self.current_mode == "READING":
            # Already in reading mode
            if self.navigation_manager.is_paused:
                self.audio_manager.speak("Already in reading mode. Navigation is paused.", force=True)
            else:
                self.audio_manager.speak("Already in reading mode.", force=True)
            return
        
        self.current_mode = "READING"
        
        # Pause navigation if active
        if self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
            self.navigation_manager.pause_navigation()
            self.audio_manager.speak("Switched to reading mode. Navigation paused.", force=True)
            print("[MODE]: Switched to READING mode (navigation paused)")
        else:
            self.audio_manager.speak("Switched to reading mode.", force=True)
            print("[MODE]: Switched to READING mode")
    
    def _switch_to_navigation_mode(self):
        """Switch to navigation mode, resuming paused navigation if available."""
        if self.current_mode == "NAVIGATION":
            # Already in navigation mode
            if self.navigation_manager.is_paused:
                self.audio_manager.speak("Already in navigation mode. Resuming navigation.", force=True)
                self.navigation_manager.resume_navigation()
            else:
                self.audio_manager.speak("Already in navigation mode.", force=True)
            return
        
        self.current_mode = "NAVIGATION"
        
        # Resume navigation if paused
        if self.navigation_manager.is_paused:
            self.navigation_manager.resume_navigation()
        elif self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
            self.audio_manager.speak("Switched to navigation mode. Continuing route.", force=True)
        else:
            self.audio_manager.speak("Switched to navigation mode. Say 'go to' followed by a destination.", force=True)
        
        print("[MODE]: Switched to NAVIGATION mode")
    
    def _toggle_mode(self):
        """Toggle between reading and navigation modes with pause/resume."""
        if self.current_mode == "NAVIGATION":
            # Switch to reading mode (pause navigation)
            self._switch_to_reading_mode()
        else:
            # Switch to navigation mode (resume if paused)
            self._switch_to_navigation_mode()
    
    def _provide_help(self):
        """Provide voice help about available commands."""
        help_text = (
            "Here are the commands you can use. "
            "Navigation: say 'go to' followed by a place, like 'go to cafeteria'. "
            "Cancel navigation by saying 'cancel navigation'. "
            "Pause by saying 'pause navigation', and resume with 'resume navigation'. "
            "Switch modes by saying 'switch mode'. "
            "Read text by saying 'read'. "
            "Describe your surroundings by saying 'describe'. "
            "Control alerts with 'mute' or 'unmute'. "
            "Ask where you are by saying 'where am I'."
        )
        self.audio_manager.speak(help_text, force=True)