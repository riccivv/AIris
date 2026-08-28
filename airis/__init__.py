"""AIris - AI assistant for visually impaired users."""

import importlib

# Lazy imports: importing the `airis` package must stay fast and lock-free.
# Eagerly importing every submodule here (esp. ui_manager -> ultralytics/torch
# and llm_manager -> openai) held module locks for tens of seconds during
# import, which deadlocked when Streamlit ran sessions/reruns concurrently.
_LAZY_MODULES = {
    'AudioManager': 'airis.audio_manager',
    'ConversationManager': 'airis.conversation_manager',
    'ConversationalVoiceHandler': 'airis.conversational_voice_handler',
    'LLMManager': 'airis.llm_manager',
    'NavigationManager': 'airis.navigation',
    'ObstacleAvoidance': 'airis.obstacle_avoidance',
    'UIManager': 'airis.ui_manager',
    'VisionAI': 'airis.vision_ai',
    'rag_lookup_destination': 'airis.knowledge_base',
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        value = getattr(importlib.import_module(_LAZY_MODULES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'airis' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY_MODULES))