import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[Error] OPENAI_API_KEY not found in .env file!")
    exit(1)

MODEL_NAME = "gpt-4.1-nano"  # Primary model
FALLBACK_MODEL = "gpt-4o-mini"  # Fallback model
MAX_TOKENS = 150
TEMPERATURE = 0.7
CACHE_ENABLED = True
CACHE_TTL = 3600

# Navigation Constants
WALKING_SPEED_MPS = 1.3

# Alert Configuration
COOLDOWN_SECONDS = 12.0
AREA_GROWTH_THRESHOLD = 0.20
HAZARD_COVERAGE_THRESHOLD = 0.06
HAZARD_HEIGHT_RATIO = 0.50

# YOLO Configuration
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE = 0.45

# New Features Configuration
ENABLE_OBSTACLE_AVOIDANCE = True
ENABLE_FACE_RECOGNITION = False
ENABLE_WEATHER_UPDATES = True
ENABLE_BATTERY_MONITORING = False

# UI Configuration
UI_THEME = 'dark'  # 'dark' or 'light'
UI_FONT_SCALE = 1.0
UI_SHOW_FPS = True
UI_SHOW_COMPASS = True
UI_ANIMATION_SPEED = 1.0

# Voice Configuration
VOICE_RATE = 160
VOICE_VOLUME = 1.0
VOICE_LANGUAGE = 'en'