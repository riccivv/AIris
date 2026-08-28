import time
import json
import os
import hashlib
from openai import OpenAI
from .config import (
    OPENAI_API_KEY,
    TEXT_MODEL,
    VISION_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    CACHE_ENABLED,
    CACHE_TTL,
    DATA_DIR,
)


ROUTER_SYSTEM_PROMPT = (
    "You are the action router for AIris, a voice assistant for visually "
    "impaired users. Choose the tool that best matches the user's spoken "
    "request and extract any needed parameters from their wording. If the "
    "utterance is ordinary conversation with no matching tool, respond with "
    "a short friendly reply instead of calling a tool."
)

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Start guidance to a named destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Place to navigate to"}
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_pending_navigation",
            "description": "Accept or reject a destination that is awaiting confirmation",
            "parameters": {
                "type": "object",
                "properties": {"accept": {"type": "boolean"}},
                "required": ["accept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Read aloud visible text such as signs or labels via the camera",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_scene",
            "description": "Describe what is currently in front of the user",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "where_am_i",
            "description": "Report the user's current location",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby",
            "description": "Find the nearest known landmark",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_navigation",
            "description": "Pause the active route",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_navigation",
            "description": "Resume a paused route",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_navigation",
            "description": "Stop navigation completely",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_alert_mute",
            "description": "Mute or unmute spoken hazard alerts",
            "parameters": {
                "type": "object",
                "properties": {"muted": {"type": "boolean"}},
                "required": ["muted"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_mode",
            "description": "Switch between READING and NAVIGATION modes",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["READING", "NAVIGATION"]}},
                "required": ["mode"],
            },
        },
    },
]


class LLMManager:
    """LLM manager with separate models for text and vision."""
    
    def __init__(self, audio_manager=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio_manager = audio_manager
        
        # Model configuration
        self.text_model = TEXT_MODEL      # "gpt-4.1-nano" for text
        self.vision_model = VISION_MODEL   # "gpt-4o-mini" for images
        self.max_tokens = MAX_TOKENS
        self.temperature = TEMPERATURE
        
        # Response cache
        self.cache = {}
        self.cache_enabled = CACHE_ENABLED
        self.cache_ttl = CACHE_TTL
        
        # Rate limiting
        self.rate_limit = 10  # Max requests per minute
        self.request_times = []
        
        # Stats
        self.total_requests = 0
        self.cache_hits = 0
        self.api_errors = 0
        self.model_usage = {}  # Track which models are used
        
        # Load cache from disk
        self.cache_file = os.path.join(DATA_DIR, "llm_cache.json")
        self.load_cache()
        
        print(f"[LLM]: Text Model: {self.text_model}")
        print(f"[LLM]: Vision Model: {self.vision_model}")
    
    def query(self, prompt, system_prompt=None, image=None, use_cache=True):
        """Main query method - automatically routes to appropriate model."""
        
        # Check rate limit
        if not self._check_rate_limit():
            return self._handle_rate_limit()
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, image)
        
        # Check cache
        if use_cache and self.cache_enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                self.cache_hits += 1
                print(f"[LLM]: Cache hit for: {prompt[:50]}...")
                return cached['response']
        
        # Make API call
        try:
            # Choose model based on whether image is present
            model = self.vision_model if image is not None else self.text_model
            print(f"[LLM]: Using {'vision' if image is not None else 'text'} model: {model}")
            
            response = self._make_api_call(prompt, system_prompt, image, model)
            
            # Cache response
            if use_cache and self.cache_enabled:
                self.cache[cache_key] = {
                    'response': response,
                    'timestamp': time.time()
                }
                self.save_cache()
            
            return response
            
        except Exception as e:
            print(f"[LLM Error]: {e}")
            return self._handle_error(e, prompt, system_prompt, image)
    
    def _make_api_call(self, prompt, system_prompt=None, image=None, model=None):
        """Make API call with the specified model."""
        self.total_requests += 1
        self.request_times.append(time.time())
        
        # Build messages
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message with optional image
        if image is not None:
            import base64
            import cv2
            
            _, buffer = cv2.imencode('.jpg', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Track model usage
            self.model_usage[model] = self.model_usage.get(model, 0) + 1
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[LLM]: Model {model} failed: {e}")
            
            # If text model fails, try vision model and vice versa
            fallback_model = self.vision_model if model == self.text_model else self.text_model
            
            try:
                print(f"[LLM]: Trying fallback model: {fallback_model}")
                response = self.client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Track model usage
                self.model_usage[fallback_model] = self.model_usage.get(fallback_model, 0) + 1
                
                return response.choices[0].message.content.strip()
                
            except Exception as fallback_error:
                print(f"[LLM]: Fallback also failed: {fallback_error}")
                raise fallback_error
    
    def _generate_cache_key(self, prompt, image=None):
        """Generate unique cache key for prompt and image."""
        key_string = prompt
        
        if image is not None:
            # Use image hash for cache key
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            key_string += f"_{image_hash}"
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_rate_limit(self):
        """Check if we're within rate limits."""
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        return len(self.request_times) < self.rate_limit
    
    def _handle_rate_limit(self):
        """Handle rate limit exceeded."""
        print("[LLM]: Rate limit exceeded")
        if self.audio_manager:
            self.audio_manager.speak("Please wait a moment before asking again.", force=True)
        return "Rate limit exceeded. Please try again in a moment."
    
    def _handle_error(self, error, prompt, system_prompt=None, image=None):
        """Handle API errors gracefully."""
        self.api_errors += 1
        
        error_messages = {
            'rate_limit_exceeded': "I'm receiving too many requests. Please try again shortly.",
            'insufficient_quota': "I'm temporarily unavailable. Please check back later.",
            'invalid_request_error': "I couldn't process that request. Could you rephrase?",
            'authentication_error': "There's an authentication issue. Please check the API key."
        }
        
        error_type = getattr(error, 'code', 'unknown')
        message = error_messages.get(error_type, "I'm having trouble. Please try again.")
        
        if self.audio_manager:
            self.audio_manager.speak(message, force=True)
        
        return message
    
    def query_text(self, prompt, system_prompt=None, use_cache=True):
        """Query with text model only."""
        return self.query(prompt, system_prompt, image=None, use_cache=use_cache)

    def route_command(self, user_input, context_note=None):
        """Route a raw utterance to an agent tool via function calling.

        Returns {'tool': name, 'args': {...}} when a tool matches,
        {'tool': 'chat'} for conversational input, or None if routing
        is unavailable (rate limit / API error).
        """
        if not self._check_rate_limit():
            print("[LLM]: Router skipped (rate limit)")
            return None

        messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        if context_note:
            messages.append({"role": "system", "content": context_note})
        messages.append({"role": "user", "content": user_input})

        try:
            self.request_times.append(time.time())
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                tools=AGENT_TOOLS,
                max_tokens=80,
                temperature=0,
            )

            message = response.choices[0].message
            self.total_requests += 1
            self.model_usage[self.text_model] = self.model_usage.get(self.text_model, 0) + 1

            if message.tool_calls:
                call = message.tool_calls[0]
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"[ROUTER]: {call.function.name} {args}")
                return {"tool": call.function.name, "args": args}

            return {"tool": "chat"}

        except Exception as e:
            print(f"[Router Error]: {e}")
            return None
    
    def query_vision(self, prompt, image, system_prompt=None, use_cache=True):
        """Query with vision model only."""
        return self.query(prompt, system_prompt, image=image, use_cache=use_cache)
    
    def get_stats(self):
        """Get LLM usage statistics."""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'api_errors': self.api_errors,
            'cache_size': len(self.cache),
            'requests_last_minute': len(self.request_times),
            'model_usage': self.model_usage,
            'text_model': self.text_model,
            'vision_model': self.vision_model
        }
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"[LLM Cache Save Error]: {e}")
    
    def load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"[LLM]: Loaded {len(self.cache)} cached responses")
        except Exception as e:
            print(f"[LLM Cache Load Error]: {e}")
    
    def clear_cache(self):
        """Clear all cached responses."""
        self.cache = {}
        self.save_cache()
        print("[LLM]: Cache cleared")
    
    def optimize_prompt(self, prompt):
        """Optimize prompt for better results and lower cost."""
        # Remove unnecessary whitespace
        prompt = " ".join(prompt.split())
        
        # Add clear instructions
        if not prompt.endswith('.'):
            prompt += '.'
        
        # Limit prompt length (reduces tokens)
        if len(prompt) > 500:
            prompt = prompt[:497] + '...'
        
        return prompt