import os
from dotenv import load_dotenv

# Paths (resolved from the project root, two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Load environment variables from the project root
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Ensure runtime directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[Error] OPENAI_API_KEY not found in .env file!")
    exit(1)

TEXT_MODEL = "gpt-4.1-nano"
VISION_MODEL = "gpt-4o-mini"
MAX_TOKENS = 150
TEMPERATURE = 0.7
CACHE_ENABLED = True
CACHE_TTL = 3600

# Navigation Constants
WALKING_SPEED_MPS = 1.3
NAV_CURVE_THRESHOLD = 100.0  # Show curved turn arrow within this many meters
NAV_TURN_THRESHOLD = 10.0    # Show solid turn arrow within this many meters

# Alert Configuration
COOLDOWN_SECONDS = 12.0
AREA_GROWTH_THRESHOLD = 0.20
HAZARD_COVERAGE_THRESHOLD = 0.06
HAZARD_HEIGHT_RATIO = 0.50

# YOLO Configuration
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolo26n.onnx')
YOLO_CONFIDENCE = 0.45

# Agent Configuration
ENABLE_LLM_ROUTING = True  # Route natural-language input to tools via LLM function calling

ENABLE_OBSTACLE_AVOIDANCE = True

# UI Configuration
UI_SHOW_FPS = True

# Voice Configuration
VOICE_RATE = 160
VOICE_VOLUME = 1.0
