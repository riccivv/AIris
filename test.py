import sys

# 1. Patch PyAudioWPatch for Python compatibility before importing speech_recognition
try:
    import pyaudiowpatch as pyaudio
    sys.modules['pyaudio'] = pyaudio
except ImportError:
    pass

import cv2
import pyttsx3
import threading
import queue
import time
import os
import base64
import speech_recognition as sr
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI

# 2. Environment & API Setup
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("[Error] OPENAI_API_KEY not found in .env file!")
    exit(1)

client = OpenAI(api_key=api_key)
model = YOLO('yolov8n.pt')

# 3. Thread-Safe Speech Queue & Acoustic Echo Suppression
speech_queue = queue.Queue()
is_speaking = False  # Mute flag to prevent mic from hearing TTS output


def _tts_worker():
    """Thread-safe worker that toggles 'is_speaking' to mute mic crosstalk and prevent freezing."""
    global is_speaking
    while True:
        item = speech_queue.get()
        if item is None:
            break
        
        # Support both raw text strings and (text, force) tuples
        if isinstance(item, tuple):
            text, force = item
        else:
            text, force = item, False

        # If alerts are globally muted AND this speech event isn't forced (e.g. Nav Cues), skip it
        if MUTE_ALERTS and not force:
            speech_queue.task_done()
            continue
        
        is_speaking = True  # Signal mic thread to ignore incoming audio
        print(f"\n[AIris]: {text}\n")
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"[TTS Worker Error]: {e}")
        finally:
            time.sleep(0.3)  # Brief buffer for speaker echo to dissipate in room
            is_speaking = False
            speech_queue.task_done()


# Start TTS worker daemon
threading.Thread(target=_tts_worker, daemon=True).start()


def speak(text, force=False):
    """Enqueues text to be spoken asynchronously. Set force=True to bypass MUTE_ALERTS."""
    speech_queue.put((text, force))


# 4. RAG Knowledge Base & Simulated Route Maps
# WALKING_SPEED = ~1.3 m/s (used to simulate walking time per step)
WALKING_SPEED_MPS = 1.3  

LOCATIONS_DB = [
    {
        "keywords": ["cafeteria", "canteen", "dining", "food"],
        "name": "Main Cafeteria",
        "details": "Ground Floor, East Wing",
        "route": [
            {"instruction": "Walk straight for 50 meters", "distance_meters": 50, "turn": "Turn right"},
            {"instruction": "Turn right in 20 meters", "distance_meters": 20, "turn": "Turn left"},
            {"instruction": "Turn left in 10 meters to enter Main Cafeteria", "distance_meters": 10, "turn": "You have arrived at Main Cafeteria"}
        ]
    },
    {
        "keywords": ["library", "study", "books"],
        "name": "Central Library",
        "details": "2nd Floor, Main Building",
        "route": [
            {"instruction": "Walk straight for 500 meters", "distance_meters": 500, "turn": "Turn left"},
            {"instruction": "Turn left in 100 meters towards the stairs", "distance_meters": 100, "turn": "Take the stairs up"},
            {"instruction": "Go up stairs and walk 30 meters", "distance_meters": 30, "turn": "You have arrived at Central Library"}
        ]
    },
    {
        "keywords": ["restroom", "toilet", "bathroom", "washroom"],
        "name": "Restroom",
        "details": "Ground Floor, West Corridor",
        "route": [
            {"instruction": "Walk straight for 30 meters", "distance_meters": 30, "turn": "Turn right"},
            {"instruction": "Turn right in 15 meters", "distance_meters": 15, "turn": "You have arrived at Restroom"}
        ]
    }
]


def rag_lookup_destination(query):
    """Retrieves matched destination object from the knowledge base using keyword fuzzy matching."""
    query_clean = query.lower()
    for item in LOCATIONS_DB:
        for kw in item["keywords"]:
            if kw in query_clean:
                return item
    return None


# 5. System State Variables
current_mode = "READING"  # Default to Reading Mode
MUTE_ALERTS = True        # OFF by default (say "unmute alerts" to activate)

pending_destination_data = None  # Destination awaiting user confirmation
current_destination_data = None  # Active route destination

# Simulated Navigation State
active_route = []
current_step_index = 0
step_start_time = 0
announced_preturn = False
announced_turn = False

last_spoken_time = {}
last_known_coverage = {}

COOLDOWN_SECONDS = 12.0       # Duration between repeated hazard alerts (12s delay)
AREA_GROWTH_THRESHOLD = 0.20  # Re-alert if object grows significantly (>20%)

notification_banner = "MODE: READING (Say 'Read', 'Describe', or 'Let's go to [place]')"
banner_expiry = time.time() + 5.0
is_processing_ai = False
latest_frame = None


def encode_frame_to_base64(frame):
    """Encodes an OpenCV image frame to base64 format for OpenAI Vision API."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def ask_openai_multimodal(prompt_text, frame):
    """Sends current frame snapshot + prompt to OpenAI GPT-4o-mini."""
    global is_processing_ai, notification_banner, banner_expiry
    if is_processing_ai or frame is None:
        return
    
    is_processing_ai = True
    speak("Processing request...", force=True)

    def _task():
        global is_processing_ai, notification_banner, banner_expiry
        try:
            cv2.imwrite("ocr_debug.jpg", frame)
            base64_image = encode_frame_to_base64(frame)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt_text}. Keep your response concise and under 25 words."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )

            text_result = response.choices[0].message.content.strip()
            speak(text_result, force=True)
            notification_banner = f"AI: {text_result[:35]}..."
            banner_expiry = time.time() + 5.0

        except Exception as e:
            speak("Failed to process request.", force=True)
            print(f"[AI Error]: {e}")
        finally:
            is_processing_ai = False

    threading.Thread(target=_task, daemon=True).start()


def cancel_navigation():
    """Cancels active navigation route and clears state."""
    global current_destination_data, pending_destination_data, active_route, current_step_index, notification_banner, banner_expiry
    current_destination_data = None
    pending_destination_data = None
    active_route = []
    current_step_index = 0
    speak("Navigation cancelled.", force=True)
    notification_banner = "Navigation Cancelled"
    banner_expiry = time.time() + 3.0


def toggle_mode(target_mode=None):
    """Switches system operational mode and resets hazard memory tracking."""
    global current_mode, last_spoken_time, last_known_coverage, notification_banner, banner_expiry
    
    if target_mode:
        current_mode = target_mode
    else:
        current_mode = "READING" if current_mode == "NAVIGATION" else "NAVIGATION"

    # Clear spatial memory on state transition
    last_spoken_time.clear()
    last_known_coverage.clear()

    speak(f"Switched to {current_mode.lower()} mode.", force=True)
    notification_banner = f"MODE: {current_mode}"
    banner_expiry = time.time() + 3.0


def start_simulated_route(destination_item):
    """Initializes time-based route simulation."""
    global active_route, current_step_index, step_start_time, announced_preturn, announced_turn
    active_route = destination_item.get("route", [])
    current_step_index = 0
    step_start_time = time.time()
    announced_preturn = False
    announced_turn = False

    if active_route:
        first_step = active_route[0]
        speak(f"Starting route to {destination_item['name']}. {first_step['instruction']}.", force=True)


def listen_for_voice_commands():
    """Background microphone worker thread with speech parsing and route cancel triggers."""
    global current_mode, is_speaking, MUTE_ALERTS, pending_destination_data, current_destination_data, notification_banner, banner_expiry
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 400
    recognizer.dynamic_energy_threshold = True

    try:
        microphone = sr.Microphone()
        print("[MIC]: Listener Running Successfully!")
    except Exception as e:
        print(f"[MIC Error]: Listener setup failed: {e}")
        return

    while True:
        if is_speaking:
            time.sleep(0.2)
            continue

        try:
            with microphone as source:
                audio = recognizer.listen(source, phrase_time_limit=4, timeout=2)
                
            if is_speaking:
                continue

            command = recognizer.recognize_google(audio).lower()
            print(f"[HEARD]: '{command}'")

            # --- CANCEL NAVIGATION TRIGGER ---
            if "cancel navigation" in command or "stop navigation" in command or "cancel route" in command or "cancel" in command:
                cancel_navigation()
                continue

            # --- CONFIRMATION FLOW FOR PENDING DESTINATION ---
            if pending_destination_data:
                if any(word in command for word in ["yes", "yeah", "confirm", "correct", "sure", "ok", "okay"]):
                    current_destination_data = pending_destination_data
                    pending_destination_data = None
                    notification_banner = f"Destination set: {current_destination_data['name']}"
                    banner_expiry = time.time() + 5.0
                    toggle_mode("NAVIGATION")
                    start_simulated_route(current_destination_data)
                    continue
                elif any(word in command for word in ["no", "stop", "wrong"]):
                    speak("Destination cancelled. Remaining in reading mode.", force=True)
                    pending_destination_data = None
                    continue

            # --- DESTINATION NAVIGATION REQUEST ("let's go to [place]") ---
            if "go to" in command or "navigate to" in command or "take me to" in command:
                target_raw = command
                for trigger in ["go to", "navigate to", "take me to"]:
                    if trigger in command:
                        target_raw = command.split(trigger)[-1].strip()
                        break

                matched_item = rag_lookup_destination(target_raw)

                if matched_item:
                    pending_destination_data = matched_item
                    speak(f"Found destination: {matched_item['name']}. Would you like to switch to navigation mode?", force=True)
                else:
                    pending_destination_data = {
                        "name": target_raw.title(),
                        "details": "Custom Location",
                        "route": [{"instruction": f"Head straight towards {target_raw.title()}", "distance_meters": 100, "turn": "You have arrived"}]
                    }
                    speak(f"Destination {target_raw} is not in the system map. Set as custom destination and proceed?", force=True)
                continue

            # --- MODE SWITCHING COMMANDS ---
            if "switch mode" in command or "toggle mode" in command or "change mode" in command:
                toggle_mode()
            elif "reading mode" in command:
                toggle_mode("READING")
            elif "navigation mode" in command or "navigate" in command:
                toggle_mode("NAVIGATION")
            
            # --- MUTE/UNMUTE HAZARD ALERTS (UNMUTE FIRST FIX) ---
            elif "unmute alert" in command or "unmute alerts" in command or "enable alerts" in command or "unmute" in command:
                MUTE_ALERTS = False
                speak("Hazard alerts unmuted.", force=True)

            elif "mute alert" in command or "mute alerts" in command or "silence alerts" in command or "mute" in command:
                MUTE_ALERTS = True
                speak("Hazard alerts muted.", force=True)

            # --- VISION AI COMMANDS ---
            elif "read" in command:
                speak("Reading text.", force=True)
                ask_openai_multimodal("Extract and read all clear visible text in this image out loud.", latest_frame)
            elif "describe" in command or "what do you see" in command:
                speak("Analyzing scene.", force=True)
                ask_openai_multimodal("Describe what is directly in front of the user concisely.", latest_frame)

        except (sr.UnknownValueError, sr.WaitTimeoutError):
            pass
        except Exception as e:
            time.sleep(0.5)


# Start Microphone Background Worker
threading.Thread(target=listen_for_voice_commands, daemon=True).start()

# Initialize Video Stream
cap = cv2.VideoCapture(0)

print("--- AIris Active ---")
print("Controls: Say 'Let's go to [place]' | Say 'Cancel Navigation' | Say 'Unmute Alerts' | Say 'Read'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame.copy()
    frame_height, frame_width, _ = frame.shape
    current_time = time.time()

    # ==========================================
    # MODE 1: NAVIGATION MODE (Route Engine + Hazard YOLO)
    # ==========================================
    if current_mode == "NAVIGATION":
        
        # --- TIME-BASED NAVIGATION ROUTE SIMULATION ---
        if active_route and current_step_index < len(active_route):
            step = active_route[current_step_index]
            total_dist = step["distance_meters"]
            turn_action = step["turn"]

            # Calculate distance covered based on elapsed walking time
            elapsed_time = current_time - step_start_time
            distance_covered = elapsed_time * WALKING_SPEED_MPS
            remaining_distance = max(0.0, total_dist - distance_covered)

            # Trigger 1: Pre-turn alert when remaining distance <= 10 meters
            if remaining_distance <= 10.0 and remaining_distance > 3.0 and not announced_preturn:
                speak(f"In 10 meters, {turn_action.lower()}.", force=True)
                announced_preturn = True

            # Trigger 2: Turn command when remaining distance <= 3 meters
            if remaining_distance <= 3.0 and not announced_turn:
                if "arrived" in turn_action.lower():
                    speak(turn_action, force=True)
                else:
                    speak(f"Please {turn_action.lower()} now.", force=True)
                announced_turn = True

            # Advance to next step when arrival threshold reached
            if remaining_distance <= 0.0:
                current_step_index += 1
                step_start_time = time.time()
                announced_preturn = False
                announced_turn = False

                if current_step_index < len(active_route):
                    next_step = active_route[current_step_index]
                    speak(next_step["instruction"], force=True)
                else:
                    speak("Route completed.", force=True)
                    active_route = []
                    current_destination_data = None

        # --- YOLO HAZARD DETECTION ---
        results = model(frame, conf=0.45, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x1 + x2) / 2
                coverage = (box_width * box_height) / (frame_width * frame_height)
                
                if center_x < frame_width / 3:
                    position = "on the left"
                elif center_x > (2 * frame_width) / 3:
                    position = "on the right"
                else:
                    position = "straight ahead"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({coverage*100:.1f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if coverage > 0.06 or (box_height / frame_height) > 0.50:
                    time_passed = current_time - last_spoken_time.get(class_name, 0)
                    prev_coverage = last_known_coverage.get(class_name, 0)
                    is_getting_closer = (coverage - prev_coverage) > AREA_GROWTH_THRESHOLD
                    
                    if time_passed > COOLDOWN_SECONDS or is_getting_closer:
                        message = f"{class_name} {position}."
                        # Hazard alerts remain subject to MUTE_ALERTS setting (force=False)
                        speak(message, force=False)
                        
                        last_spoken_time[class_name] = current_time
                        last_known_coverage[class_name] = coverage
                        notification_banner = f"[MUTED] {message}" if MUTE_ALERTS else message
                        banner_expiry = current_time + 2.5

    # ==========================================
    # MODE 2: READING MODE (Default)
    # ==========================================
    elif current_mode == "READING":
        margin_x, margin_y = int(frame_width * 0.15), int(frame_height * 0.15)
        cv2.rectangle(frame, (margin_x, margin_y), 
                      (frame_width - margin_x, frame_height - margin_y), (255, 165, 0), 2)
        cv2.putText(frame, "SAY 'READ' TO PROCESS TEXT", (margin_x + 10, margin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # UI Header
    header_color = (0, 255, 0) if current_mode == "NAVIGATION" else (255, 165, 0)
    cv2.rectangle(frame, (0, 0), (frame_width, 35), (30, 30, 30), -1)
    
    mute_status_str = " (ALERTS OFF)" if MUTE_ALERTS else ""
    cv2.putText(frame, f"MODE: [{current_mode}]{mute_status_str} (Press 'M' or Say 'Switch Mode')", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2)

    # UI Notification Banner
    if current_time < banner_expiry:
        cv2.rectangle(frame, (0, frame_height - 35), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.putText(frame, f"ALERT: {notification_banner}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # UI Bottom Center Destination & Route Card
    dest_item = pending_destination_data if pending_destination_data else current_destination_data
    if dest_item:
        dest_name = dest_item["name"]
        
        if pending_destination_data:
            dest_text = f"DESTINATION: {dest_name} (Confirm? Say 'Yes' / 'No')"
        elif active_route and current_step_index < len(active_route):
            rem_d = max(0, int(active_route[current_step_index]["distance_meters"] - ((current_time - step_start_time) * WALKING_SPEED_MPS)))
            dest_text = f"NAV: {dest_name} | {active_route[current_step_index]['turn']} in {rem_d}m"
        else:
            dest_text = f"DESTINATION: {dest_name}"

        text_size = cv2.getTextSize(dest_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
        card_w, card_h = text_size[0] + 20, text_size[1] + 16
        card_x = (frame_width - card_w) // 2
        card_y = frame_height - 50

        card_color = (0, 140, 255) if pending_destination_data else (0, 180, 0)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), card_color, 2)
        cv2.putText(frame, dest_text, (card_x + 10, card_y + card_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("AIris Live Field-of-View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('m'), ord('M')):
        toggle_mode()

cap.release()
cv2.destroyAllWindows()