import speech_recognition as sr
import threading
import time
from conversation_manager import ConversationManager


class ConversationalVoiceHandler:
    """Enhanced voice handler with conversational capabilities."""
    
    def __init__(self, audio_manager, navigation_manager, vision_ai, location_tracker=None):
        self.audio_manager = audio_manager
        self.navigation_manager = navigation_manager
        self.vision_ai = vision_ai
        self.location_tracker = location_tracker
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(audio_manager, vision_ai)
        
        self.current_mode = "READING"
        self.pending_destination = None
        self.latest_frame = None
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        
        # Conversation state
        self.conversation_active = False
        self.last_command_time = time.time()
        self.conversation_timeout = 30  # Return to command mode after 30s of silence
        
        # Start listener thread
        threading.Thread(target=self._listen_loop, daemon=True).start()
    
    def _listen_loop(self):
        """Background microphone worker thread."""
        try:
            microphone = sr.Microphone()
            print("[MIC]: Conversational Listener Running!")
        except Exception as e:
            print(f"[MIC Error]: {e}")
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
                self._process_input(command)
                
                # Update conversation state
                current_time = time.time()
                if current_time - self.last_command_time > self.conversation_timeout:
                    self.conversation_active = False
                
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                pass
            except Exception as e:
                time.sleep(0.5)
    
    def _process_input(self, user_input):
        """Process user input - either as command or conversation."""
        self.last_command_time = time.time()
        
        # Check for conversation activation keywords
        if self._should_activate_conversation(user_input):
            self.conversation_active = True
            self.conversation_manager.conversation_active = True
        
        # If conversation is active, try conversation first
        if self.conversation_active:
            # Check for command overrides
            if self._is_command(user_input):
                self._process_command(user_input)
            else:
                # Process as conversation
                context_update = {
                    'current_mode': self.current_mode,
                    'current_destination': self.navigation_manager.current_destination.get('name') 
                        if self.navigation_manager.current_destination else None
                }
                self.conversation_manager.process_conversation(
                    user_input, 
                    self.latest_frame,
                    context_update
                )
        else:
            # Normal command mode
            self._process_command(user_input)
    
    def _should_activate_conversation(self, user_input):
        """Determine if conversation should be activated."""
        activation_phrases = [
            'let\'s talk', 'talk to me', 'chat', 'conversation',
            'tell me about', 'what do you think', 'how are you',
            'i want to talk', 'can we talk', 'let\'s chat'
        ]
        return any(phrase in user_input.lower() for phrase in activation_phrases)
    
    def _is_command(self, user_input):
        """Check if input is a system command rather than conversation."""
        command_keywords = [
            'cancel navigation', 'stop navigation', 'go to', 'navigate to',
            'reading mode', 'navigation mode', 'switch mode', 'mute', 'unmute',
            'read', 'describe', 'where am i', 'what\'s near'
        ]
        return any(keyword in user_input.lower() for keyword in command_keywords)
    
    def _process_command(self, command):
        """Process system commands (existing functionality)."""
        # Cancel navigation
        if any(word in command for word in ["cancel navigation", "stop navigation", "cancel route", "cancel"]):
            self.navigation_manager.cancel_navigation()
            self.pending_destination = None
            return
        
        # Confirmation flow
        if self.pending_destination:
            if any(word in command for word in ["yes", "yeah", "confirm", "correct", "sure", "ok", "okay"]):
                destination = self.pending_destination
                self.pending_destination = None
                self.current_mode = "NAVIGATION"
                self.navigation_manager.start_route(destination)
                return
            elif any(word in command for word in ["no", "stop", "wrong"]):
                self.audio_manager.speak("Destination cancelled.", force=True)
                self.pending_destination = None
                return
        
        # Location queries
        if any(phrase in command for phrase in ["where am i", "current location", "my location"]):
            if self.location_tracker:
                description = self.location_tracker.get_location_description()
                self.audio_manager.speak(description, force=True)
            return
        
        # Navigation requests
        if any(trigger in command for trigger in ["go to", "navigate to", "take me to"]):
            from knowledge_base import rag_lookup_destination
            
            for trigger in ["go to", "navigate to", "take me to"]:
                if trigger in command:
                    target_raw = command.split(trigger)[-1].strip()
                    break
            
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
        
        # Mode switching
        if "reading mode" in command:
            self.current_mode = "READING"
            self.audio_manager.speak("Switched to reading mode.", force=True)
        elif "navigation mode" in command:
            self.current_mode = "NAVIGATION"
            self.audio_manager.speak("Switched to navigation mode.", force=True)
        elif "switch mode" in command or "toggle mode" in command:
            self.current_mode = "READING" if self.current_mode == "NAVIGATION" else "NAVIGATION"
            self.audio_manager.speak(f"Switched to {self.current_mode.lower()} mode.", force=True)
        
        # Mute/unmute
        elif "unmute" in command:
            self.audio_manager.set_mute_alerts(False)
            self.audio_manager.speak("Alerts unmuted.", force=True)
        elif "mute" in command:
            self.audio_manager.set_mute_alerts(True)
            self.audio_manager.speak("Alerts muted.", force=True)
        
        # Vision AI commands
        elif "read" in command:
            self.audio_manager.speak("Reading text.", force=True)
            self.vision_ai.ask_multimodal(
                "Extract and read all visible text clearly.", 
                self.latest_frame
            )
        elif "describe" in command or "what do you see" in command:
            self.audio_manager.speak("Analyzing scene.", force=True)
            self.vision_ai.ask_multimodal(
                "Describe what's directly in front of the user concisely.", 
                self.latest_frame
            )