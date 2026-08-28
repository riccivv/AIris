import speech_recognition as sr
import threading
import time
from .config import ENABLE_LLM_ROUTING
from .conversation_manager import ConversationManager


class ConversationalVoiceHandler:
    """Enhanced voice handler with conversational capabilities."""
    
    def __init__(self, audio_manager, navigation_manager, vision_ai, location_tracker=None, llm_manager=None):
        self.audio_manager = audio_manager
        self.navigation_manager = navigation_manager
        self.vision_ai = vision_ai
        self.location_tracker = location_tracker
        
        # Initialize conversation manager with shared LLM manager
        if llm_manager:
            self.conversation_manager = ConversationManager(
                audio_manager, 
                vision_ai,
                llm_manager
            )
        else:
            self.conversation_manager = ConversationManager(
                audio_manager, 
                vision_ai
            )
        
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
        self.conversation_timeout = 30
        
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
                
                self._process_input(command)
                
                current_time = time.time()
                if current_time - self.last_command_time > self.conversation_timeout:
                    self.conversation_active = False
                
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                pass
            except Exception as e:
                time.sleep(0.5)
    
    def _process_input(self, user_input):
        """Process user input with command priority."""
        self.last_command_time = time.time()

        # Keyword fast-path first: instant, free, and works offline
        if self._is_command(user_input):
            print(f"[COMMAND]: Detected command: '{user_input}'")
            self._process_command(user_input)
            return

        # Check for conversation activation
        if self._should_activate_conversation(user_input):
            self.conversation_active = True
            self.conversation_manager.conversation_active = True
            self.audio_manager.speak("I'm listening. What would you like to talk about?", force=True)
            return

        # Check for quick responses
        quick_response = self._handle_quick_responses(user_input)
        if quick_response:
            self.audio_manager.speak(quick_response, force=True)
            return

        # LLM tool routing catches natural phrasing the keywords missed
        if self.llm_manager and ENABLE_LLM_ROUTING:
            context_note = None
            if self.pending_destination:
                context_note = (
                    f"'{self.pending_destination['name']}' is awaiting navigation "
                    "confirmation; the user may be accepting or rejecting it."
                )
            decision = self.llm_manager.route_command(user_input, context_note)
            if decision:
                if decision["tool"] == "chat":
                    # Router judged it conversational - treat as implicit activation
                    self.conversation_active = True
                    self.conversation_manager.conversation_active = True
                else:
                    print(f"[ROUTED]: '{user_input}' -> {decision['tool']}")
                    self._dispatch_tool(decision["tool"], decision.get("args", {}))
                    return

        # Process as conversation if active
        if self.conversation_active:
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
            self._handle_unrecognized_input(user_input)
    
    def _is_command(self, user_input):
        """Check if input is a system command."""
        command_keywords = [
            'go to', 'navigate to', 'take me to', 'cancel navigation', 
            'stop navigation', 'cancel route', 'pause navigation', 'resume navigation',
            'reading mode', 'navigation mode', 'switch mode', 'toggle mode',
            'mute', 'unmute', 'read', 'describe', 'what do you see',
            'where am i', 'current location', 'my location', 
            "what's near", 'nearby', 'yes', 'no', 'confirm', 'cancel',
            'help', 'what can i say'
        ]
        
        user_lower = user_input.lower().strip()
        
        for keyword in command_keywords:
            if keyword in user_lower:
                # Special handling for short words to avoid false positives
                if keyword in ['read', 'describe', 'mute', 'unmute', 'yes', 'no', 'cancel', 'help']:
                    if keyword == 'read':
                        return user_lower in ['read', 'read this', 'read text', 'read it']
                    elif keyword == 'describe':
                        return user_lower in ['describe', 'describe scene', 'describe this']
                    elif keyword in ['mute', 'unmute']:
                        if keyword in user_lower.split():
                            return True
                        continue  # e.g. 'mute' inside 'unmute' - keep checking other keywords
                    elif keyword in ['yes', 'no']:
                        return user_lower == keyword
                    elif keyword == 'cancel':
                        return 'cancel' in user_lower.split()
                    elif keyword == 'help':
                        return user_lower in ['help', 'help me']
                else:
                    return True
        
        return False
    
    def _should_activate_conversation(self, user_input):
        """Determine if conversation should be activated."""
        activation_phrases = [
            "let's talk", 'talk to me', 'chat', 'conversation',
            'tell me about', 'what do you think', 'how are you',
            'i want to talk', 'can we talk', "let's chat"
        ]
        return any(phrase in user_input.lower() for phrase in activation_phrases)
    
    def _handle_quick_responses(self, user_input):
        """Handle quick responses."""
        greeting = self.conversation_manager.handle_greeting(user_input)
        if greeting:
            return greeting
        
        thanks = self.conversation_manager.handle_thanks(user_input)
        if thanks:
            return thanks
        
        goodbye = self.conversation_manager.handle_goodbye(user_input)
        if goodbye:
            self.conversation_active = False
            return goodbye
        
        return None
    
    def _handle_unrecognized_input(self, user_input):
        """Handle unrecognized input."""
        if '?' in user_input or user_input.startswith(('what', 'how', 'can', 'could', 'would')):
            hint = "You can say 'Go to [place]' for navigation, 'Read' to read text, or 'Let's talk' for conversation."
            self.audio_manager.speak(hint, force=True)
    
    def _process_command(self, command):
        """Process system commands via keyword dispatch."""
        user_lower = command.lower().strip()

        if any(word in user_lower for word in ["cancel navigation", "stop navigation", "cancel route"]):
            self._cmd_cancel_navigation()
            return

        if any(word in user_lower for word in ["pause navigation", "pause route"]):
            self._cmd_pause_navigation()
            return

        if any(word in user_lower for word in ["resume navigation", "resume route", "continue navigation"]):
            self._cmd_resume_navigation()
            return

        # CONFIRMATION FLOW
        if self.pending_destination:
            if any(word in user_lower for word in ["yes", "yeah", "confirm", "correct", "sure", "ok", "okay"]):
                self._confirm_pending()
                return
            elif any(word in user_lower for word in ["no", "stop", "wrong", "cancel"]):
                self._deny_pending()
                return

        # DESTINATION NAVIGATION REQUEST
        if any(trigger in user_lower for trigger in ["go to", "navigate to", "take me to"]):
            target_raw = user_lower
            for trigger in ["go to", "navigate to", "take me to"]:
                if trigger in user_lower:
                    target_raw = user_lower.split(trigger)[-1].strip()
                    break
            self._start_navigation_flow(target_raw)
            return

        # LOCATION QUERIES
        if any(phrase in user_lower for phrase in ["where am i", "current location", "my location"]):
            self._cmd_where_am_i()
            return

        if any(phrase in user_lower for phrase in ["what's near", "what is near", "nearby", "what's around"]):
            self._cmd_find_nearby()
            return

        # MODE SWITCHING
        if "reading mode" in user_lower:
            self._switch_to_reading_mode()
            return
        elif "navigation mode" in user_lower or user_lower == "navigate":
            self._switch_to_navigation_mode()
            return
        elif "switch mode" in user_lower or "toggle mode" in user_lower:
            self._toggle_mode()
            return

        # MUTE/UNMUTE
        if any(word in user_lower for word in ["unmute", "enable alerts"]):
            self._cmd_set_mute(False)
            return
        elif any(word in user_lower for word in ["mute", "silence alerts"]):
            self._cmd_set_mute(True)
            return

        # VISION AI COMMANDS
        if user_lower in ["read", "read this", "read text", "read it"]:
            self._cmd_read_text()
            return
        elif any(phrase in user_lower for phrase in ["describe", "what do you see"]):
            self._cmd_describe_scene()
            return

        # HELP
        if "help" in user_lower or "what can i say" in user_lower:
            self._provide_help()
            return

    def _dispatch_tool(self, name, args):
        """Execute an LLM-routed tool by name."""
        if name == "navigate_to":
            self._start_navigation_flow(str(args.get("destination", "")).strip())
        elif name == "confirm_pending_navigation":
            if args.get("accept"):
                self._confirm_pending()
            else:
                self._deny_pending()
        elif name == "read_text":
            self._cmd_read_text()
        elif name == "describe_scene":
            self._cmd_describe_scene()
        elif name == "where_am_i":
            self._cmd_where_am_i()
        elif name == "find_nearby":
            self._cmd_find_nearby()
        elif name == "pause_navigation":
            self._cmd_pause_navigation()
        elif name == "resume_navigation":
            self._cmd_resume_navigation()
        elif name == "cancel_navigation":
            self._cmd_cancel_navigation()
        elif name == "set_alert_mute":
            self._cmd_set_mute(bool(args.get("muted", False)))
        elif name == "switch_mode":
            if str(args.get("mode", "")).upper() == "NAVIGATION":
                self._switch_to_navigation_mode()
            else:
                self._switch_to_reading_mode()
        else:
            self.audio_manager.speak("I'm not sure how to do that yet.", force=True)

    def _start_navigation_flow(self, target_raw):
        """Resolve a destination and ask for confirmation before routing."""
        if not target_raw:
            self.audio_manager.speak("Where would you like to go?", force=True)
            return

        from .knowledge_base import rag_lookup_destination

        matched_item = rag_lookup_destination(target_raw)

        if matched_item:
            # Check if already at destination
            if self.location_tracker and self.location_tracker.is_at_location(matched_item['name']):
                self.audio_manager.speak(
                    f"You are already at {matched_item['name']}. No navigation needed.",
                    force=True
                )
                return

            self.pending_destination = matched_item
            self.conversation_active = False
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

    def _confirm_pending(self):
        destination = self.pending_destination
        self.pending_destination = None
        self.current_mode = "NAVIGATION"
        self.navigation_manager.start_route(destination)

    def _deny_pending(self):
        self.audio_manager.speak("Destination cancelled.", force=True)
        self.pending_destination = None

    def _cmd_cancel_navigation(self):
        self.navigation_manager.cancel_navigation()
        self.pending_destination = None
        self.conversation_active = False

    def _cmd_pause_navigation(self):
        if self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
            self.navigation_manager.pause_navigation()
            self.audio_manager.speak("Navigation paused.", force=True)
        else:
            self.audio_manager.speak("No active navigation to pause.", force=True)

    def _cmd_resume_navigation(self):
        if self.navigation_manager.is_paused:
            self.navigation_manager.resume_navigation()
        else:
            self.audio_manager.speak("Navigation is not paused.", force=True)

    def _cmd_where_am_i(self):
        if self.location_tracker:
            description = self.location_tracker.get_location_description()
            self.audio_manager.speak(description, force=True)
        else:
            self.audio_manager.speak("Location tracking is not available.", force=True)

    def _cmd_find_nearby(self):
        if self.location_tracker:
            nearest = self.location_tracker.get_nearest_landmark()
            if nearest:
                self.audio_manager.speak(
                    f"The nearest landmark is {nearest['name']}, about {nearest['distance']:.1f} meters away.",
                    force=True
                )
            else:
                self.audio_manager.speak("No known landmarks nearby.", force=True)
        else:
            self.audio_manager.speak("Location tracking is not available.", force=True)

    def _cmd_set_mute(self, muted):
        self.audio_manager.set_mute_alerts(muted)
        self.audio_manager.speak("Alerts muted." if muted else "Alerts unmuted.", force=True)

    def _cmd_read_text(self):
        self.audio_manager.speak("Reading text.", force=True)
        self.vision_ai.ask_multimodal(
            "Extract and read all visible text clearly.",
            self.latest_frame
        )

    def _cmd_describe_scene(self):
        self.audio_manager.speak("Analyzing scene.", force=True)
        self.vision_ai.ask_multimodal(
            "Describe what's directly in front of the user concisely.",
            self.latest_frame
        )
    
    def _provide_help(self):
        """Provide voice help."""
        help_text = (
            "Here are the commands you can use. "
            "Navigation: say 'go to' followed by a place, like 'go to cafeteria'. "
            "Cancel navigation by saying 'cancel navigation'. "
            "Pause by saying 'pause navigation', and resume with 'resume navigation'. "
            "Switch modes by saying 'switch mode'. "
            "Read text by saying 'read'. "
            "Describe your surroundings by saying 'describe'. "
            "Control alerts with 'mute' or 'unmute'. "
            "Ask where you are by saying 'where am I'. "
            "You can also just speak naturally - for example, 'could you read this sign for me?'"
        )
        self.audio_manager.speak(help_text, force=True)
    
    def _switch_to_reading_mode(self):
        """Switch to reading mode."""
        if self.current_mode == "READING":
            self.audio_manager.speak("Already in reading mode.", force=True)
            return
        
        self.current_mode = "READING"
        
        if self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
            self.navigation_manager.pause_navigation()
            self.audio_manager.speak("Switched to reading mode. Navigation paused.", force=True)
        else:
            self.audio_manager.speak("Switched to reading mode.", force=True)
    
    def _switch_to_navigation_mode(self):
        """Switch to navigation mode."""
        if self.current_mode == "NAVIGATION":
            if self.navigation_manager.is_paused:
                self.audio_manager.speak("Resuming navigation.", force=True)
                self.navigation_manager.resume_navigation()
            else:
                self.audio_manager.speak("Already in navigation mode.", force=True)
            return
        
        self.current_mode = "NAVIGATION"
        
        if self.navigation_manager.is_paused:
            self.navigation_manager.resume_navigation()
        elif self.navigation_manager.navigation_started and not self.navigation_manager.route_completed:
            self.audio_manager.speak("Switched to navigation mode. Continuing route.", force=True)
        else:
            self.audio_manager.speak("Switched to navigation mode.", force=True)
    
    def _toggle_mode(self):
        """Toggle between modes."""
        if self.current_mode == "NAVIGATION":
            self._switch_to_reading_mode()
        else:
            self._switch_to_navigation_mode()