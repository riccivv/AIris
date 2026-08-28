# AIris - AI Assistant for Visually Impaired Users

AIris is an AI-powered assistant designed to help visually impaired users navigate their environment, read text, and interact with their surroundings using voice commands and computer vision.

## Features

- 🎯 **Real-time Object Detection**: Uses YOLO26 to detect obstacles, pedestrians, and vehicles
- 🗣️ **Voice Commands**: Hands-free operation with speech recognition
- 📖 **Text Reading**: Reads text from the environment using AI
- 🧭 **Navigation**: Turn-by-turn navigation with pause/resume capability
- 🤖 **Conversational AI**: Natural conversation with context awareness
- ⚠️ **Hazard Alerts**: Real-time alerts for nearby obstacles
- **Location Tracking**: Simulated dead-reckoning positioning with nearby landmarks

## Installation

### Prerequisites

- Python 3.8+
- Webcam
- Microphone
- OpenAI API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AIris.git
cd AIris
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run

```bash
# Streamlit web app (recommended)
streamlit run app.py
# or on Windows
run_streamlit.bat

# Terminal CLI variant
python main.py
```

### Config

The app needs an OpenAI API key in a `.env` file:
```ini
OPENAI_API_KEY=sk-...
```