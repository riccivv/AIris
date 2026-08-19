import threading
import time
import json
import os
import random
from llm_manager import LLMManager


class ConversationManager:
    """Manages conversational dialogue with context awareness and LLM integration."""
    
    def __init__(self, audio_manager, vision_ai=None):
        self.llm_manager = LLMManager(audio_manager)
        self.audio_manager = audio_manager
        self.vision_ai = vision_ai
        
        # Conversation memory
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges for context
        self.context = {
            'user_name': None,
            'current_mode': 'READING',
            'current_destination': None,
            'last_topic': None,
            'mood': 'neutral'
        }
        
        # Conversation state
        self.is_processing = False
        self.conversation_active = False
        self.last_interaction = time.time()
        
        # Load conversation history if exists
        self.load_history()
    
    def process_conversation(self, user_input, frame=None, context_update=None):
        """Process conversational input and generate response using LLM Manager."""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # Update context if provided
        if context_update:
            self.context.update(context_update)
        
        def _task():
            try:
                # Build system prompt with context
                system_prompt = self._build_system_prompt()
                
                # Add conversation history to prompt
                history_prompt = self._build_history_prompt()
                
                # Combine prompts
                if history_prompt:
                    full_prompt = f"{history_prompt}\nUser: {user_input}"
                else:
                    full_prompt = user_input
                
                # Determine if image should be included
                include_image = frame is not None and self._should_include_image(user_input)
                
                # Use LLM Manager with caching and fallback
                response = self.llm_manager.query(
                    full_prompt,
                    system_prompt=system_prompt,
                    image=frame if include_image else None,
                    use_cache=True
                )
                
                # Update conversation history
                self._update_history(user_input, response)
                
                # Extract and update context
                self._update_context(user_input, response)
                
                # Speak the response
                self.audio_manager.speak(response, force=True)
                
                # Save conversation
                self.save_history()
                
            except Exception as e:
                print(f"[Conversation Error]: {e}")
                self.audio_manager.speak("I'm having trouble processing that. Could you repeat?", force=True)
            finally:
                self.is_processing = False
                self.last_interaction = time.time()
        
        threading.Thread(target=_task, daemon=True).start()
    
    def _build_system_prompt(self):
        """Build system prompt with context awareness."""
        prompt = """You are AIris, a friendly and helpful AI assistant for visually impaired users.
        You help with:
        1. Reading text and describing scenes
        2. Navigation and directions
        3. General conversation and companionship
        4. Answering questions about the environment
        
        Keep responses:
        - Clear and concise (under 30 words for voice responses)
        - Warm and friendly in tone
        - Appropriate for audio (no visual descriptions like "as you can see")
        - Aware of the user's current context and mode
        """
        
        # Add user-specific context
        if self.context['user_name']:
            prompt += f"\nThe user's name is {self.context['user_name']}. "
        
        prompt += f"\nCurrent mode: {self.context['current_mode']}. "
        
        if self.context['current_destination']:
            prompt += f"Currently navigating to: {self.context['current_destination']}. "
        
        if self.context['mood'] != 'neutral':
            prompt += f"The user seems {self.context['mood']}. "
        
        if self.context['last_topic']:
            prompt += f"Last topic discussed: {self.context['last_topic']}. "
        
        return prompt
    
    def _build_history_prompt(self):
        """Build prompt from recent conversation history."""
        if not self.conversation_history:
            return ""
        
        history_lines = []
        # Use last 5 exchanges for context
        for exchange in self.conversation_history[-5:]:
            history_lines.append(f"User: {exchange['user']}")
            history_lines.append(f"AIris: {exchange['assistant']}")
        
        return "\n".join(history_lines)
    
    def _should_include_image(self, user_input):
        """Determine if image should be included in context."""
        visual_keywords = [
            'what do you see', 'describe', 'look', 'read', 'around',
            'nearby', 'here', 'this', 'color', 'shape', 'object',
            'where', 'environment', 'surroundings', 'front', 'see',
            'view', 'watch', 'looking', 'visible'
        ]
        return any(keyword in user_input.lower() for keyword in visual_keywords)
    
    def _update_history(self, user_input, ai_response):
        """Update conversation history."""
        self.conversation_history.append({
            'user': user_input,
            'assistant': ai_response,
            'timestamp': time.time()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _update_context(self, user_input, ai_response):
        """Extract and update context from conversation."""
        user_lower = user_input.lower()
        
        # Detect name introduction
        if 'my name is' in user_lower:
            name = user_lower.split('my name is')[-1].strip().rstrip('.').title()
            if name and len(name) < 30:  # Sanity check
                self.context['user_name'] = name
        
        # Detect mood
        if any(word in user_lower for word in ['happy', 'great', 'good', 'wonderful', 'excited']):
            self.context['mood'] = 'positive'
        elif any(word in user_lower for word in ['sad', 'tired', 'frustrated', 'angry', 'upset']):
            self.context['mood'] = 'negative'
        elif any(word in user_lower for word in ['okay', 'fine', 'alright']):
            self.context['mood'] = 'neutral'
        
        # Track topics
        if 'weather' in user_lower:
            self.context['last_topic'] = 'weather'
        elif 'news' in user_lower:
            self.context['last_topic'] = 'news'
        elif 'time' in user_lower:
            self.context['last_topic'] = 'time'
        elif 'navigation' in user_lower or 'direction' in user_lower:
            self.context['last_topic'] = 'navigation'
    
    def handle_greeting(self, user_input):
        """Handle greetings and small talk."""
        greetings = {
            'hello': "Hello! How are you today?",
            'hi': "Hi there! How can I help you?",
            'hey': "Hey! What can I do for you?",
            'good morning': "Good morning! I hope you're having a great day.",
            'good afternoon': "Good afternoon! How can I assist you?",
            'good evening': "Good evening! What do you need help with?",
            'how are you': "I'm doing well, thank you for asking! How are you?",
            "what's up": "Not much, just ready to help you! What's on your mind?",
            'who are you': "I'm AIris, your AI assistant. I can help you navigate, read text, and describe your surroundings.",
            'what can you do': "I can help you with navigation, reading text, describing scenes, and general conversation. Just ask!",
        }
        
        user_lower = user_input.lower().strip()
        
        # Check for partial matches
        for greeting, response in greetings.items():
            if greeting in user_lower:
                return response
        
        return None
    
    def handle_thanks(self, user_input):
        """Handle expressions of gratitude."""
        thanks_responses = [
            "You're welcome!",
            "Happy to help!",
            "Anytime!",
            "My pleasure!",
            "Glad I could assist!"
        ]
        
        if any(word in user_input.lower() for word in ['thank', 'thanks', 'appreciate']):
            return random.choice(thanks_responses)
        
        return None
    
    def handle_goodbye(self, user_input):
        """Handle farewell messages."""
        goodbyes = {
            'bye': "Goodbye! Stay safe!",
            'goodbye': "Goodbye! I'm here if you need me.",
            'see you': "See you later! Take care!",
            'good night': "Good night! Sweet dreams!",
        }
        
        user_lower = user_input.lower().strip()
        
        for goodbye, response in goodbyes.items():
            if goodbye in user_lower:
                return response
        
        return None
    
    def get_status(self):
        """Get conversation status."""
        return {
            'active': self.conversation_active,
            'last_interaction': self.last_interaction,
            'history_length': len(self.conversation_history),
            'context': self.context,
            'llm_stats': self.llm_manager.get_stats()
        }
    
    def get_stats(self):
        """Get LLM statistics."""
        return self.llm_manager.get_stats()
    
    def save_history(self, filename='conversation_history.json'):
        """Save conversation history to file."""
        try:
            data = {
                'history': self.conversation_history,
                'context': self.context,
                'saved_at': time.time()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Conversation Save Error]: {e}")
    
    def load_history(self, filename='conversation_history.json'):
        """Load conversation history from file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('history', [])
                    self.context.update(data.get('context', {}))
                    print(f"[Conversation]: Loaded {len(self.conversation_history)} previous exchanges")
        except Exception as e:
            print(f"[Conversation Load Error]: {e}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.llm_manager.clear_cache()
        self.save_history()
        self.audio_manager.speak("Conversation history and cache cleared.", force=True)