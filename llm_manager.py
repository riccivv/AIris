import time
import json
import os
import hashlib
import threading
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME, FALLBACK_MODEL, MAX_TOKENS, TEMPERATURE


class LLMManager:
    """Production-ready LLM manager with caching, fallbacks, and optimization."""
    
    def __init__(self, audio_manager=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.audio_manager = audio_manager
        
        # Configuration
        self.primary_model = MODEL_NAME  # "gpt-4.1-nano"
        self.fallback_model = FALLBACK_MODEL  # "gpt-4o-mini"
        self.max_tokens = MAX_TOKENS
        self.temperature = TEMPERATURE
        
        # Check available models on initialization
        self.available_models = self._check_available_models()
        
        # Response cache
        self.cache = {}
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 hour cache lifetime
        
        # Rate limiting
        self.rate_limit = 10  # Max requests per minute
        self.request_times = []
        
        # Stats
        self.total_requests = 0
        self.cache_hits = 0
        self.api_errors = 0
        self.model_usage = {}  # Track which models are used
        
        # Load cache from disk
        self.cache_file = "llm_cache.json"
        self.load_cache()
        
        print(f"[LLM]: Primary model: {self.primary_model}")
        print(f"[LLM]: Fallback model: {self.fallback_model}")
        if self.primary_model in self.available_models:
            print(f"[LLM]: Primary model available ✓")
        else:
            print(f"[LLM]: Primary model not available, will use fallback")
    
    def _check_available_models(self):
        """Check which models are available in the API."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"[LLM]: Could not fetch available models: {e}")
            return []
    
    def query(self, prompt, system_prompt=None, image=None, use_cache=True):
        """Main query method with caching and fallback."""
        
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
            response = self._make_api_call(prompt, system_prompt, image)
            
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
    
    def _make_api_call(self, prompt, system_prompt=None, image=None):
        """Make API call with proper error handling and model fallback."""
        self.total_requests += 1
        self.request_times.append(time.time())
        
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
        
        # Determine which models to try
        models_to_try = []
        
        # Check if primary model is available
        if self.primary_model in self.available_models or not self.available_models:
            models_to_try.append(self.primary_model)
        
        # Add fallback model
        if self.fallback_model != self.primary_model:
            models_to_try.append(self.fallback_model)
        
        # Try each model
        last_error = None
        for model in models_to_try:
            try:
                print(f"[LLM]: Trying model: {model}")
                
                # Check if model supports images
                if image is not None and "nano" in model:
                    # gpt-4.1-nano might not support images
                    # Convert to text-only description
                    messages = self._convert_to_text_only(messages)
                
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
                last_error = e
                print(f"[LLM]: Model {model} failed: {e}")
                continue
        
        # If all models failed
        raise last_error if last_error else Exception("All models failed")
    
    def _convert_to_text_only(self, messages):
        """Convert image messages to text-only for models that don't support images."""
        converted_messages = []
        
        for message in messages:
            if isinstance(message.get('content'), list):
                # Extract just the text parts
                text_parts = []
                for part in message['content']:
                    if part['type'] == 'text':
                        text_parts.append(part['text'])
                    elif part['type'] == 'image_url':
                        text_parts.append("[Image attached - describe what you see]")
                
                converted_messages.append({
                    'role': message['role'],
                    'content': ' '.join(text_parts)
                })
            else:
                converted_messages.append(message)
        
        return converted_messages
    
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
    
    def query_with_context(self, prompt, context=None, image=None):
        """Query with context injection."""
        if context:
            context_text = "\n".join([f"- {item}" for item in context])
            full_prompt = f"Context:\n{context_text}\n\nUser Query: {prompt}"
        else:
            full_prompt = prompt
        
        return self.query(full_prompt, image=image)
    
    def batch_query(self, prompts, system_prompt=None):
        """Process multiple queries efficiently."""
        results = []
        
        for prompt in prompts:
            result = self.query(prompt, system_prompt)
            results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def stream_query(self, prompt, system_prompt=None, image=None, callback=None):
        """Stream response token by token."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.primary_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            full_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    
                    if callback:
                        callback(token)
            
            return full_response
            
        except Exception as e:
            print(f"[LLM Stream Error]: {e}")
            return self._handle_error(e, prompt, system_prompt)
    
    def get_stats(self):
        """Get LLM usage statistics."""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'api_errors': self.api_errors,
            'cache_size': len(self.cache),
            'requests_last_minute': len(self.request_times),
            'model_usage': self.model_usage,
            'primary_model': self.primary_model,
            'fallback_model': self.fallback_model,
            'primary_available': self.primary_model in self.available_models
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