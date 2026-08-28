import pyttsx3
import threading
import queue
import time
import sys
from .config import VOICE_RATE, VOICE_VOLUME

# Patch PyAudioWPatch for Python compatibility
try:
    import pyaudiowpatch as pyaudio
    sys.modules['pyaudio'] = pyaudio
except ImportError:
    pass


class AudioManager:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.mute_alerts = False  # Alerts enabled by default; mute via voice command
        
        # Start TTS worker daemon
        threading.Thread(target=self._tts_worker, daemon=True).start()
    
    def _tts_worker(self):
        """Thread-safe worker that toggles 'is_speaking' to mute mic crosstalk."""
        while True:
            item = self.speech_queue.get()
            if item is None:
                break
            
            # Support both raw text strings and (text, force) tuples
            if isinstance(item, tuple):
                text, force = item
            else:
                text, force = item, False

            # If alerts are globally muted AND this speech event isn't forced, skip it
            if self.mute_alerts and not force:
                self.speech_queue.task_done()
                continue
            
            self.is_speaking = True
            print(f"\n[AIris]: {text}\n")
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', VOICE_RATE)
                engine.setProperty('volume', VOICE_VOLUME)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS Worker Error]: {e}")
            finally:
                time.sleep(0.3)  # Brief buffer for speaker echo
                self.is_speaking = False
                self.speech_queue.task_done()
    
    def speak(self, text, force=False):
        """Enqueues text to be spoken asynchronously."""
        self.speech_queue.put((text, force))
    
    def set_mute_alerts(self, muted):
        """Set whether hazard alerts should be muted."""
        self.mute_alerts = muted