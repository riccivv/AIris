import cv2
import base64
import threading
import time
import os
from .config import DATA_DIR
from .llm_manager import LLMManager


class VisionAI:
    def __init__(self, audio_manager, llm_manager=None):
        self.audio_manager = audio_manager
        
        # Use provided LLM manager or create new one
        if llm_manager:
            self.llm_manager = llm_manager
        else:
            self.llm_manager = LLMManager(audio_manager)
        
        self.is_processing = False
        self.latest_result = None
    
    def ask_multimodal(self, prompt_text, frame, notification_callback=None):
        """Sends current frame snapshot + prompt to vision model."""
        if self.is_processing or frame is None:
            return
        
        self.is_processing = True
        # No spoken "Processing request..." - the UI shows a typing indicator
        # while is_processing is True; only the final result is spoken.

        def _task():
            try:
                # Save debug image
                cv2.imwrite(os.path.join(DATA_DIR, "ocr_debug.jpg"), frame)
                
                # Build system prompt for vision tasks
                system_prompt = """You are AIris, an AI assistant for visually impaired users.
                Analyze the image and provide clear, concise descriptions.
                Keep responses under 25 words for voice output."""
                
                # Use vision model (automatically selected by LLM Manager)
                response = self.llm_manager.query_vision(
                    prompt_text,
                    frame,
                    system_prompt=system_prompt
                )
                
                # Speak the response
                self.audio_manager.speak(response, force=True)
                
                # Update notification if callback provided
                if notification_callback:
                    notification_callback(f"AI: {response[:35]}...", 5.0)
                
                print(f"[VISION AI]: {response}")
                
            except Exception as e:
                self.audio_manager.speak("Failed to process request.", force=True)
                print(f"[AI Error]: {e}")
            finally:
                self.is_processing = False

        threading.Thread(target=_task, daemon=True).start()
    
    def read_text(self, frame, notification_callback=None):
        """Read text from image."""
        prompt = "Extract and read all clear visible text in this image out loud."
        self.ask_multimodal(prompt, frame, notification_callback)
    
    def describe_scene(self, frame, notification_callback=None):
        """Describe the scene in front of the user."""
        prompt = "Describe what is directly in front of the user concisely."
        self.ask_multimodal(prompt, frame, notification_callback)
    
    def get_stats(self):
        """Get LLM statistics."""
        if self.llm_manager:
            return self.llm_manager.get_stats()
        return {}