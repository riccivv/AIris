import streamlit as st
import cv2
import time
import threading
import numpy as np
from PIL import Image
import speech_recognition as sr
import queue
import os
import logging
import html as _html

# Suppress warnings
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Import modules
from airis.audio_manager import AudioManager
from airis.navigation import NavigationManager
from airis.vision_ai import VisionAI
from airis.conversation_manager import ConversationManager
from airis.llm_manager import LLMManager
from airis.location.base_tracker import LocationTracker
from airis.knowledge_base import rag_lookup_destination
from airis.ui_manager import draw_navigation_arrow, UIManager
from airis.config import *

# Page config
st.set_page_config(page_title="AIris", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.audio_manager = None
    st.session_state.navigation_manager = None
    st.session_state.vision_ai = None
    st.session_state.llm_manager = None
    st.session_state.conversation_manager = None
    st.session_state.location_tracker = None
    st.session_state.ui_manager = None
    st.session_state.current_mode = "READING"
    st.session_state.conversation_active = False
    st.session_state.messages = []
    st.session_state.camera_active = False
    st.session_state.mic_active = False
    st.session_state.camera = None
    st.session_state.mic = None
    st.session_state.command_queue = queue.Queue()
    st.session_state.is_speaking = False  # Track AI speaking state

# CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 40px;
        margin: 2px 0;
        padding: 0 10px;
        font-size: 14px;
        border-radius: 4px;
    }
    .main-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
        padding: 5px 0;
    }
    .section-header {
        font-size: 16px;
        font-weight: bold;
        margin: 15px 0 5px 0;
        padding-bottom: 3px;
        border-bottom: 2px solid #444444;
    }
    .custom-divider {
        border-top: 2px solid #444444;
        margin: 10px 0;
    }
    /* Slightly raise content while keeping clearance below the header bar */
    [data-testid="stMainBlockContainer"], .block-container {
        padding-top: 3.5rem;
    }
    /* Fit everything on one screen without scrolling */
    [data-testid="stMain"] { height: 100vh; overflow: hidden; }
    [data-testid="stMainBlockContainer"] {
        height: 100vh;
        overflow: hidden;
        padding-bottom: 0;
        max-width: 100%;
    }
/* Chat container ends above the "Type a message" input instead of being covered by it */
    .st-key-chat_box {
        height: calc(100vh - 120px) !important;
        padding-bottom: 28px;
    }
    /* Cap the camera image so it never pushes the page below the fold */
    [data-testid="stImage"] img {
        max-height: calc(100vh - 220px);
        object-fit: contain;
        width: 100%;
    }
    /* Messenger-style typing indicator */
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #7ee0a3;
        display: inline-block;
        animation: typing-bounce 1.2s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-6px); opacity: 1; }
    }
    /* AIris branding: pure CSS (no re-created DOM node, so no flicker on rerun) */
    [data-testid="stHeader"]::before {
        content: "";
        position: fixed;
        top: 13px;
        left: calc(50% - 69px);
        width: 54px;
        height: 54px;
        background-image: url("data:image/png;base64,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");
        background-repeat: no-repeat;
        background-size: contain;
        z-index: 2147483647;
        pointer-events: none;
    }
    [data-testid="stHeader"]::after {
        content: "Iris";
        position: fixed;
        top: 9px;
        left: calc(50% + 32px);
        transform: translateX(-50%);
        z-index: 2147483647;
        font-size: 62px;
        font-weight: bold;
        color: #ffffff;
        line-height: 1;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

class ContinuousCamera:
    def __init__(self, ui_manager=None):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.latest_frame = None
        self.raw_frame = None
        self.frame_lock = threading.Lock()
        self.current_mode = "READING"  # Local mode tracking
        self.navigation_started = False
        self.navigation_paused = False
        self.navigation_step = 0
        self.navigation_total = 0
        self.conversation_active = False
        self.navigation_manager = None
        self.ui_manager = ui_manager  # Provides detect_hazards (YOLO + spoken alerts)
        self.detected_frame = None    # Last frame processed by the detection thread
        self.detected_objects = []  # Latest detection results (boxes for live overlay)
        self.detection_thread = None
    
    def start(self):
        """Start camera capture."""
        for backend, index in [(cv2.CAP_DSHOW, 0), (cv2.CAP_MSMF, 0), (cv2.CAP_ANY, 0)]:
            try:
                self.cap = cv2.VideoCapture(index, backend)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"[CAMERA]: Success with backend {backend}")
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for less lag
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from the camera
                        break
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"[CAMERA]: Backend {backend} failed: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None
        
        if self.cap is None or not self.cap.isOpened():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        return True
    
    def update_status(self):
        """Update status from session state (called from main thread)."""
        try:
            self.current_mode = st.session_state.get('current_mode', 'READING')
            self.conversation_active = st.session_state.get('conversation_active', False)
            self.navigation_manager = st.session_state.get('navigation_manager', None)
            
            if st.session_state.navigation_manager:
                status = st.session_state.navigation_manager.get_status()
                self.navigation_started = status.get('active', False)
                self.navigation_paused = status.get('paused', False)
                self.navigation_step = status.get('current_step', 0)
                self.navigation_total = status.get('total_steps', 0)
        except:
            pass
    
    def _capture_loop(self):
        """Continuous capture loop. Never blocks on YOLO inference."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Store raw frame
                with self.frame_lock:
                    self.raw_frame = frame.copy()
                
                # In navigation mode, overlay the latest detection boxes on the
                # live frame so the feed stays smooth while inference runs slower
                if self.current_mode == "NAVIGATION" and self.detected_objects:
                    self._draw_detections(frame)
                
                # Add overlays
                frame = self._add_overlays(frame)
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store display frame
                with self.frame_lock:
                    self.latest_frame = frame_rgb
                
                # Calculate FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
            else:
                time.sleep(0.01)
            
            time.sleep(0.02)  # allow up to 50 FPS capture (camera limits it to ~30)

    def _detection_loop(self):
        """Run YOLO hazard detection in the background so the capture loop never blocks.

        Processes the latest raw frame at a modest rate and stores the annotated
        result in self.detected_frame for the capture loop to display.
        """
        while self.running and self.cap and self.cap.isOpened():
            with self.frame_lock:
                frame = self.raw_frame
            if (frame is not None
                    and self.current_mode == "NAVIGATION"
                    and self.ui_manager is not None
                    and self.ui_manager.model is not None):
                try:
                    annotated = frame.copy()
                    detected = self.ui_manager.detect_hazards(annotated)  # returns object list
                    self.detected_objects = detected
                    self.detected_frame = annotated
                except Exception as e:
                    print(f"[Hazard Detection Error]: {e}")
            time.sleep(0.5)  # ~2 FPS detection to keep CPU free for the live feed

    def _draw_detections(self, frame):
        """Draw the latest detection boxes on a live frame (cheap, no inference)."""
        for d in self.detected_objects:
            x1, y1, x2, y2 = d.get('bbox', (0, 0, 0, 0))
            label = f"{d.get('class', '?')} ({d.get('distance', '?')})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def _add_overlays(self, frame):
        """Add UI overlays to frame."""
        frame_height, frame_width = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (frame_width, 30), (30, 30, 30), -1)
        
        mode = self.current_mode
        color = (0, 255, 0) if mode == "NAVIGATION" else (255, 165, 0)
        cv2.putText(frame, f"MODE: {mode}", (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # FPS
        cv2.putText(frame, f"{self.fps} FPS", (frame_width - 60, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Navigation status
        if self.navigation_started:
            status_text = "PAUSED" if self.navigation_paused else f"STEP {self.navigation_step+1}/{self.navigation_total}"
            status_color = (0, 165, 255) if self.navigation_paused else (0, 255, 0)
            
            cv2.rectangle(frame, (frame_width - 120, 40), (frame_width - 10, 60), (30, 30, 30), -1)
            cv2.putText(frame, status_text, (frame_width - 110, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            # Directional arrow guidance
            try:
                if getattr(self, 'navigation_manager', None):
                    draw_navigation_arrow(frame, self.navigation_manager)
            except Exception as e:
                print(f"[Arrow Error]: {e}")

            # Destination card with remaining distance and step progress
            try:
                if self.ui_manager is not None:
                    self.ui_manager.draw_destination_card(frame)
            except Exception as e:
                print(f"[Destination Card Error]: {e}")
        
        # Conversation indicator
        if self.conversation_active:
            cv2.putText(frame, "CONV", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame
    
    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame
    
    def get_raw_frame(self):
        with self.frame_lock:
            return self.raw_frame
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.detection_thread:
            self.detection_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        self.cap = None

class ContinuousMicrophone:
    def __init__(self, command_queue, audio_manager=None):
        self.command_queue = command_queue
        self.audio_manager = audio_manager
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        self.running = False
        self.thread = None
        self.microphone = None
        
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            st.session_state.mic_error = f"Mic error: {e}"
    
    def start(self):
        if self.microphone:
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            return True
        return False
    
    def _ai_speaking(self):
        return bool(self.audio_manager and self.audio_manager.is_speaking)
    
    def _listen_loop(self):
        while self.running:
            # Check if AI is speaking - if so, don't listen
            if self._ai_speaking():
                time.sleep(0.2)
                continue
            
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, phrase_time_limit=5, timeout=1)
                
                # Double-check AI is not speaking before processing
                if self._ai_speaking():
                    continue
                
                command = self.recognizer.recognize_google(audio)
                if command:
                    self.command_queue.put(command)
                
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class CustomAudioManager(AudioManager):
    """Audio manager that speaks reliably from a worker thread and never touches
    st.session_state from the worker thread."""
    
    def __init__(self):
        super().__init__()
        self.spoken_history = []   # Every line actually spoken (for chat transcript)
        self._spoken_lock = threading.Lock()
        self._synced_count = 0
    
    def _tts_worker(self):
        """Thread-safe worker that speaks queued text via SAPI directly.

        pyttsx3's runAndWait() only emits audio for the first utterance on
        Windows (known bug), so we use SAPI.SpVoice directly through win32com,
        which speaks synchronously and reliably on every call.
        """
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass
        
        import win32com.client
        
        while True:
            item = self.speech_queue.get()
            if item is None:
                break
            
            if isinstance(item, tuple):
                text, force = item
            else:
                text, force = item, False
            
            if self.mute_alerts and not force:
                self.speech_queue.task_done()
                continue
            
            self.is_speaking = True
            print(f"\n[AIris]: {text}\n")
            with self._spoken_lock:
                self.spoken_history.append(text)
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Rate = max(-10, min(10, int((VOICE_RATE - 200) / 20)))
                voice.Volume = max(0, min(100, int(VOICE_VOLUME * 100)))
                voice.Speak(text)
            except Exception as e:
                print(f"[TTS Error]: {e}")
            finally:
                time.sleep(0.3)
                self.is_speaking = False
                self.speech_queue.task_done()

def initialize_system():
    if not st.session_state.initialized:
        with st.spinner("Initializing..."):
            try:
                st.session_state.audio_manager = CustomAudioManager()
                st.session_state.llm_manager = LLMManager(st.session_state.audio_manager)
                st.session_state.navigation_manager = NavigationManager(st.session_state.audio_manager)
                st.session_state.vision_ai = VisionAI(st.session_state.audio_manager, st.session_state.llm_manager)
                st.session_state.conversation_manager = ConversationManager(
                    st.session_state.audio_manager,
                    st.session_state.vision_ai,
                    st.session_state.llm_manager
                )
                st.session_state.location_tracker = LocationTracker()
                st.session_state.navigation_manager.set_location_tracker(st.session_state.location_tracker)
                st.session_state.ui_manager = UIManager(
                    st.session_state.audio_manager,
                    st.session_state.navigation_manager
                )
                st.session_state.initialized = True
                
                # TTS self-test: confirms voice output works and logs engine status
                st.session_state.audio_manager.speak("AIris is ready.", force=True)
            except Exception as e:
                st.error(f"Init failed: {e}")

def get_raw_frame():
    if st.session_state.camera_active and st.session_state.camera:
        return st.session_state.camera.get_raw_frame()
    return None


def start_navigation_to(target_raw):
    """Resolve a destination and start the route. Returns a chat response or None."""
    matched_item = rag_lookup_destination(target_raw)
    if not matched_item:
        return "Destination not found."
    if st.session_state.location_tracker.is_at_location(matched_item['name']):
        return f"You are already at {matched_item['name']}."
    st.session_state.navigation_manager.start_route(matched_item)
    st.session_state.current_mode = "NAVIGATION"
    return None  # start_route already speaks the route start


def dispatch_routed_tool(tool, args):
    """Execute an LLM-routed tool against session managers.

    Returns a chat response string, or None when the tool handles its
    own spoken output asynchronously.
    """
    if tool == "navigate_to":
        return start_navigation_to(str(args.get("destination", "")).strip())

    elif tool == "read_text":
        raw_frame = get_raw_frame()
        if raw_frame is None:
            return "No frame available. Please start the camera first."
        st.session_state.vision_ai.read_text(raw_frame)
        return None

    elif tool == "describe_scene":
        raw_frame = get_raw_frame()
        if raw_frame is None:
            return "No frame available. Please start the camera first."
        st.session_state.vision_ai.describe_scene(raw_frame)
        return None

    elif tool == "where_am_i":
        return st.session_state.location_tracker.get_location_description()

    elif tool == "find_nearby":
        nearest = st.session_state.location_tracker.get_nearest_landmark()
        if nearest:
            return f"The nearest landmark is {nearest['name']}, about {nearest['distance']:.1f} meters away."
        return "No known landmarks nearby."

    elif tool == "pause_navigation":
        st.session_state.navigation_manager.pause_navigation()
        return "Navigation paused."

    elif tool == "resume_navigation":
        st.session_state.navigation_manager.resume_navigation()
        return "Navigation resumed."

    elif tool == "cancel_navigation":
        st.session_state.navigation_manager.cancel_navigation()
        st.session_state.current_mode = "READING"
        return "Navigation cancelled."

    elif tool == "set_alert_mute":
        muted = bool(args.get("muted", False))
        st.session_state.audio_manager.set_mute_alerts(muted)
        return "Alerts muted." if muted else "Alerts unmuted."

    elif tool == "switch_mode":
        mode = str(args.get("mode", "")).upper()
        if mode == "NAVIGATION" and st.session_state.current_mode != "NAVIGATION":
            st.session_state.current_mode = "NAVIGATION"
            if st.session_state.navigation_manager and st.session_state.navigation_manager.is_paused:
                st.session_state.navigation_manager.resume_navigation()
            return "Navigation mode."
        elif mode != "NAVIGATION" and st.session_state.current_mode != "READING":
            st.session_state.current_mode = "READING"
            if st.session_state.navigation_manager and st.session_state.navigation_manager.navigation_started:
                st.session_state.navigation_manager.pause_navigation()
            return "Reading mode."
        return None

    elif tool == "confirm_pending_navigation":
        return "Nothing to confirm right now."

    return "I couldn't do that just yet."


def process_command(command):
    user_lower = command.lower().strip()
    st.session_state.messages.append({"role": "user", "content": command})
    
    # Navigation commands
    if any(trigger in user_lower for trigger in ["go to", "navigate to", "take me to"]):
        for trigger in ["go to", "navigate to", "take me to"]:
            if trigger in user_lower:
                target_raw = user_lower.split(trigger)[-1].strip()
                break

        response = start_navigation_to(target_raw)
    
    elif "cancel navigation" in user_lower:
        st.session_state.navigation_manager.cancel_navigation()
        st.session_state.current_mode = "READING"
        response = "Navigation cancelled."
    
    elif "pause navigation" in user_lower:
        st.session_state.navigation_manager.pause_navigation()
        response = "Navigation paused."
    
    elif "resume navigation" in user_lower:
        st.session_state.navigation_manager.resume_navigation()
        response = "Navigation resumed."
    
    elif "reading mode" in user_lower:
        st.session_state.current_mode = "READING"
        response = "Reading mode."
    
    elif "navigation mode" in user_lower:
        st.session_state.current_mode = "NAVIGATION"
        response = "Navigation mode."
    
    elif "switch mode" in user_lower:
        if st.session_state.current_mode == "READING":
            st.session_state.current_mode = "NAVIGATION"
            if st.session_state.navigation_manager and st.session_state.navigation_manager.is_paused:
                st.session_state.navigation_manager.resume_navigation()
            response = "Navigation mode."
        else:
            st.session_state.current_mode = "READING"
            if st.session_state.navigation_manager and st.session_state.navigation_manager.navigation_started:
                st.session_state.navigation_manager.pause_navigation()
            response = "Reading mode."
    
    elif "read" in user_lower and user_lower in ["read", "read this", "read text", "read it"]:
        raw_frame = None
        if st.session_state.camera_active and st.session_state.camera:
            raw_frame = st.session_state.camera.get_raw_frame()
        
        if raw_frame is not None:
            st.session_state.vision_ai.read_text(raw_frame)
            response = None  # Result is spoken by the vision thread; typing indicator shows meanwhile
        else:
            response = "No frame available. Please start the camera first."
    
    elif "describe" in user_lower or "what do you see" in user_lower:
        raw_frame = None
        if st.session_state.camera_active and st.session_state.camera:
            raw_frame = st.session_state.camera.get_raw_frame()
        
        if raw_frame is not None:
            st.session_state.vision_ai.describe_scene(raw_frame)
            response = None  # Result is spoken by the vision thread; typing indicator shows meanwhile
        else:
            response = "No frame available. Please start the camera first."
    
    elif "conversation mode" in user_lower:
        if "off" in user_lower:
            st.session_state.conversation_active = False
            response = "Conversation mode off."
        else:
            st.session_state.conversation_active = True
            response = "Conversation mode on."
    
    elif "where am i" in user_lower:
        response = st.session_state.location_tracker.get_location_description()
    
    elif any(word in user_lower for word in ["unmute", "enable alerts"]):
        st.session_state.audio_manager.set_mute_alerts(False)
        response = "Alerts unmuted."
    
    elif any(word in user_lower for word in ["mute", "silence alerts"]):
        st.session_state.audio_manager.set_mute_alerts(True)
        response = "Alerts muted."
    
    else:
        # LLM tool routing catches natural phrasing the keywords missed
        decision = None
        if ENABLE_LLM_ROUTING and st.session_state.llm_manager is not None:
            decision = st.session_state.llm_manager.route_command(user_lower)

        if decision and decision["tool"] != "chat":
            print(f"[ROUTED]: '{user_lower}' -> {decision['tool']}")
            response = dispatch_routed_tool(decision["tool"], decision.get("args", {}))
        else:
            if decision and decision["tool"] == "chat":
                st.session_state.conversation_active = True

            if st.session_state.conversation_active:
                context_update = {
                    'current_mode': st.session_state.current_mode,
                    'current_destination': st.session_state.navigation_manager.current_destination.get('name')
                        if st.session_state.navigation_manager.current_destination else None
                }
                st.session_state.conversation_manager.process_conversation(
                    user_lower,
                    get_raw_frame(),
                    context_update
                )
                response = None  # Real response spoken by the conversation thread; typing indicator shows meanwhile
            else:
                response = st.session_state.conversation_manager.handle_greeting(user_lower)
                if not response:
                    response = "How can I help?"
    
    if response:
        st.session_state.audio_manager.speak(response, force=True)

def sync_spoken_to_chat():
    """Append everything the audio manager has spoken into the chat transcript."""
    audio = st.session_state.audio_manager
    if not audio or not hasattr(audio, 'spoken_history'):
        return
    with audio._spoken_lock:
        history = list(audio.spoken_history)
    for text in history[audio._synced_count:]:
        st.session_state.messages.append({"role": "assistant", "content": text})
    audio._synced_count = len(history)

def _render_chat_messages():
    """Render chat history as styled chat bubbles (user right, AIris left)."""
    if not st.session_state.messages:
        st.info("No messages yet")
        return
    
    for message in st.session_state.messages:
        role = message.get('role', 'assistant')
        content = _html.escape(str(message.get('content', '')))
        
        if role == 'user':
            align = 'flex-end'
            direction = 'row-reverse'
            avatar = '🙂'
            name = 'You'
            bubble_bg = '#2b3a67'
            name_color = '#8ab4f8'
        else:
            align = 'flex-start'
            direction = 'row'
            avatar = '🤖'
            name = 'AIris'
            bubble_bg = '#1f2a44'
            name_color = '#7ee0a3'
        
        st.markdown(
            f"""
            <div style="display:flex; justify-content:{align}; margin:8px 0;">
                <div style="display:flex; flex-direction:{direction}; align-items:flex-end; max-width:80%;">
                    <div style="width:30px; height:30px; border-radius:50%; background:#3a4a6b;
                                display:flex; align-items:center; justify-content:center;
                                font-size:16px; flex-shrink:0;">{avatar}</div>
                    <div style="background:{bubble_bg}; border-radius:14px; padding:8px 12px; margin:0 8px;
                                word-wrap:break-word;">
                        <div style="font-size:11px; font-weight:bold; color:{name_color}; margin-bottom:2px;">{name}</div>
                        <div style="font-size:14px; color:#eaeaea;">{content}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Typing indicator while AIris is processing (vision/conversation task running)
    vision = st.session_state.get('vision_ai')
    conversation = st.session_state.get('conversation_manager')
    if (vision and getattr(vision, 'is_processing', False)) or (conversation and getattr(conversation, 'is_processing', False)):
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
                <div style="display:flex; flex-direction:row; align-items:flex-end; max-width:80%;">
                    <div style="width:30px; height:30px; border-radius:50%; background:#3a4a6b;
                                display:flex; align-items:center; justify-content:center;
                                font-size:16px; flex-shrink:0;">🤖</div>
                    <div style="background:#1f2a44; border-radius:14px; padding:12px 14px; margin:0 8px;
                                display:flex; gap:5px;">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    initialize_system()
    sync_spoken_to_chat()
    
    # Update camera status from main thread
    if st.session_state.camera_active and st.session_state.camera:
        st.session_state.camera.update_status()
    
    # Advance navigation state (drives remaining distance, step changes, turn alerts)
    nav = st.session_state.navigation_manager
    if nav and nav.navigation_started and not nav.route_completed:
        try:
            nav.update(time.time())
        except Exception as e:
            print(f"[Navigation Update Error]: {e}")
    
    # Always return to reading mode once a route finishes
    if nav and nav.route_completed:
        st.session_state.current_mode = "READING"
        nav.route_completed = False
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">Input Devices</div>', unsafe_allow_html=True)
        
        cam_col, mic_col = st.columns(2, gap="small")
        
        with cam_col:
            camera_label = "Stop Camera" if st.session_state.camera_active else "Start Camera"
            camera_type = "primary" if st.session_state.camera_active else "secondary"
            
            if st.button(camera_label, key="camera_btn", type=camera_type, width="stretch"):
                if not st.session_state.camera_active:
                    st.session_state.camera = ContinuousCamera(st.session_state.ui_manager)
                    if st.session_state.camera.start():
                        st.session_state.camera_active = True
                        st.rerun()
                    else:
                        st.error("Failed to open camera")
                else:
                    st.session_state.camera.stop()
                    st.session_state.camera_active = False
                    st.session_state.camera = None
                    st.rerun()
        
        with mic_col:
            mic_label = "Stop Mic" if st.session_state.mic_active else "Start Mic"
            mic_type = "primary" if st.session_state.mic_active else "secondary"
            
            if st.button(mic_label, key="mic_btn", type=mic_type, width="stretch"):
                if not st.session_state.mic_active:
                    st.session_state.mic = ContinuousMicrophone(
                        st.session_state.command_queue, st.session_state.audio_manager)
                    if st.session_state.mic.start():
                        st.session_state.mic_active = True
                    else:
                        st.error("Mic failed")
                else:
                    st.session_state.mic.stop()
                    st.session_state.mic_active = False
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Mode buttons
        st.markdown('<div class="section-header">Mode</div>', unsafe_allow_html=True)
        
        mode_col1, mode_col2 = st.columns(2, gap="small")
        
        with mode_col1:
            if st.button("Reading", key="reading_mode", 
                        type="primary" if st.session_state.current_mode == "READING" else "secondary",
                        width="stretch"):
                st.session_state.current_mode = "READING"
                if st.session_state.navigation_manager and st.session_state.navigation_manager.navigation_started:
                    st.session_state.navigation_manager.pause_navigation()
                st.session_state.audio_manager.speak("Reading mode.", force=True)
        
        with mode_col2:
            if st.button("Navigation", key="nav_mode",
                        type="primary" if st.session_state.current_mode == "NAVIGATION" else "secondary",
                        width="stretch"):
                st.session_state.current_mode = "NAVIGATION"
                if st.session_state.navigation_manager and st.session_state.navigation_manager.is_paused:
                    st.session_state.navigation_manager.resume_navigation()
                st.session_state.audio_manager.speak("Navigation mode.", force=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Conversation toggle
        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)
        
        if st.button("Conversation Mode: ON" if st.session_state.conversation_active else "Conversation Mode: OFF",
                    key="conversation_btn",
                    type="primary" if st.session_state.conversation_active else "secondary",
                    width="stretch"):
            st.session_state.conversation_active = not st.session_state.conversation_active
            st.session_state.audio_manager.speak(
                "Conversation mode on." if st.session_state.conversation_active else "Conversation mode off.",
                force=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        
        if st.button("Read Text", width="stretch"):
            st.session_state.messages.append({"role": "user", "content": "Read Text"})
            if st.session_state.camera_active and st.session_state.camera:
                raw_frame = st.session_state.camera.get_raw_frame()
                if raw_frame is not None:
                    st.session_state.vision_ai.read_text(raw_frame)
                else:
                    st.warning("No frame available")
            else:
                st.warning("Start camera first")
        
        if st.button("Describe Scene", width="stretch"):
            st.session_state.messages.append({"role": "user", "content": "Describe Scene"})
            if st.session_state.camera_active and st.session_state.camera:
                raw_frame = st.session_state.camera.get_raw_frame()
                if raw_frame is not None:
                    st.session_state.vision_ai.describe_scene(raw_frame)
                else:
                    st.warning("No frame available")
            else:
                st.warning("Start camera first")
    
    # Main area
    main_col1, main_col2 = st.columns([3, 2])
    
    with main_col1:
        st.markdown('<div class="section-header">Camera Feed</div>', unsafe_allow_html=True)
        
        camera_placeholder = st.empty()
        
        if st.session_state.camera_active and st.session_state.camera:
            frame = st.session_state.camera.get_frame()
            if frame is not None:
                camera_placeholder.image(frame, channels="RGB", width="stretch")
            else:
                camera_placeholder.info("Starting camera...")
        else:
            camera_placeholder.info("Camera is off. Use sidebar to start.")
    
    with main_col2:
        st.markdown('<div class="section-header">Chat</div>', unsafe_allow_html=True)
        
        chat_container = st.container(height=480, key="chat_box")
        
        with chat_container:
            _render_chat_messages()
    
    # Text chat input (docked at the bottom of the page)
    user_text = st.chat_input("Type a message for AIris...")
    if user_text:
        process_command(user_text)
    
    # Process voice commands
    try:
        while not st.session_state.command_queue.empty():
            command = st.session_state.command_queue.get_nowait()
            process_command(command)
    except:
        pass
    
    # Auto-refresh
    time.sleep(0.033)  # ~30 FPS display refresh
    st.rerun()

if __name__ == "__main__":
    main()