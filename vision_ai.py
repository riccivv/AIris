import cv2
import base64
import threading
from openai import OpenAI
from config import OPENAI_API_KEY


class VisionAI:
    def __init__(self, audio_manager):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio_manager = audio_manager
        self.is_processing = False
        self.latest_result = None
    
    def encode_frame_to_base64(self, frame):
        """Encodes an OpenCV image frame to base64 format."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def ask_multimodal(self, prompt_text, frame, notification_callback=None):
        """Sends current frame snapshot + prompt to OpenAI GPT-4o-mini."""
        if self.is_processing or frame is None:
            return
        
        self.is_processing = True
        self.audio_manager.speak("Processing request...", force=True)

        def _task():
            try:
                cv2.imwrite("ocr_debug.jpg", frame)
                base64_image = self.encode_frame_to_base64(frame)

                response = self.client.chat.completions.create(
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
                self.audio_manager.speak(text_result, force=True)
                
                if notification_callback:
                    notification_callback(f"AI: {text_result[:35]}...", 5.0)

            except Exception as e:
                self.audio_manager.speak("Failed to process request.", force=True)
                print(f"[AI Error]: {e}")
            finally:
                self.is_processing = False

        threading.Thread(target=_task, daemon=True).start()