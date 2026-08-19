import time
import math
import json
import os
from urllib.parse import quote


class MapIntegration:
    """Map integration for location display and navigation."""
    
    def __init__(self):
        self.map_provider = os.getenv('MAP_PROVIDER', 'openstreetmap')  # or 'google'
        self.api_key = os.getenv('MAPS_API_KEY', '')
        self.cache_dir = 'map_cache'
        self.cache_duration = 3600  # 1 hour cache
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_map_url(self, position, zoom=18):
        """Generate map URL for current position."""
        lat, lon = position.get('y', 0), position.get('x', 0)
        
        if self.map_provider == 'openstreetmap':
            return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}"
        elif self.map_provider == 'google':
            return f"https://www.google.com/maps?q={lat},{lon}&z={zoom}"
        else:
            return f"https://maps.google.com/?q={lat},{lon}"
    
    def get_static_map(self, position, size=(640, 480), zoom=18):
        """Get static map image for current position."""
        lat, lon = position.get('y', 0), position.get('x', 0)
        
        if self.map_provider == 'openstreetmap':
            return self._get_osm_static_map(lat, lon, size, zoom)
        elif self.map_provider == 'google' and self.api_key:
            return self._get_google_static_map(lat, lon, size, zoom)
        else:
            return None
    
    def _get_osm_static_map(self, lat, lon, size, zoom):
        """Get static map from OpenStreetMap."""
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            
            # Calculate tile coordinates
            tile_size = 256
            scale = 2 ** zoom
            xtile = int((lon + 180) / 360 * scale)
            ytile = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * scale)
            
            # Build tile URL
            url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
            
            # Check cache
            cache_file = os.path.join(self.cache_dir, f"map_{zoom}_{xtile}_{ytile}.png")
            
            if os.path.exists(cache_file):
                if time.time() - os.path.getmtime(cache_file) < self.cache_duration:
                    return cv2.imread(cache_file) if 'cv2' in globals() else Image.open(cache_file)
            
            # Download tile
            response = requests.get(url, headers={'User-Agent': 'AIris/1.0'})
            
            if response.status_code == 200:
                # Save to cache
                with open(cache_file, 'wb') as f:
                    f.write(response.content)
                
                # Convert to image
                img = Image.open(BytesIO(response.content))
                return img
                
        except Exception as e:
            print(f"[Map Error]: {e}")
        
        return None
    
    def _get_google_static_map(self, lat, lon, size, zoom):
        """Get static map from Google Maps."""
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            
            url = (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}"
                f"&markers=color:red%7C{lat},{lon}&key={self.api_key}"
            )
            
            response = requests.get(url)
            
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
                
        except Exception as e:
            print(f"[Google Maps Error]: {e}")
        
        return None
    
    def get_directions(self, start, end, mode='walking'):
        """Get directions between two points."""
        if self.map_provider == 'openstreetmap':
            return self._get_osm_directions(start, end, mode)
        elif self.map_provider == 'google' and self.api_key:
            return self._get_google_directions(start, end, mode)
        
        return None
    
    def _get_osm_directions(self, start, end, mode='walking'):
        """Get directions from OpenStreetMap (OSRM)."""
        try:
            import requests
            
            # OSRM API
            url = (
                f"https://router.project-osrm.org/route/v1/{mode}/"
                f"{start[0]},{start[1]};{end[0]},{end[1]}"
                f"?overview=full&geometries=geojson&steps=true"
            )
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('routes'):
                    route = data['routes'][0]
                    return {
                        'distance': route['distance'],
                        'duration': route['duration'],
                        'steps': route['legs'][0]['steps']
                    }
                    
        except Exception as e:
            print(f"[Directions Error]: {e}")
        
        return None
    
    def _get_google_directions(self, start, end, mode='walking'):
        """Get directions from Google Maps."""
        try:
            import requests
            
            url = (
                f"https://maps.googleapis.com/maps/api/directions/json?"
                f"origin={start[0]},{start[1]}&destination={end[0]},{end[1]}"
                f"&mode={mode}&key={self.api_key}"
            )
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('routes'):
                    route = data['routes'][0]['legs'][0]
                    return {
                        'distance': route['distance']['value'],
                        'duration': route['duration']['value'],
                        'steps': route['steps']
                    }
                    
        except Exception as e:
            print(f"[Google Directions Error]: {e}")
        
        return None
    
    def geocode(self, address):
        """Convert address to coordinates."""
        if self.map_provider == 'openstreetmap':
            return self._geocode_osm(address)
        elif self.map_provider == 'google' and self.api_key:
            return self._geocode_google(address)
        
        return None
    
    def _geocode_osm(self, address):
        """Geocode using OpenStreetMap Nominatim."""
        try:
            import requests
            
            url = (
                f"https://nominatim.openstreetmap.org/search?"
                f"q={quote(address)}&format=json&limit=1"
            )
            
            response = requests.get(url, headers={'User-Agent': 'AIris/1.0'})
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        'lat': float(data[0]['lat']),
                        'lon': float(data[0]['lon']),
                        'display_name': data[0]['display_name']
                    }
                    
        except Exception as e:
            print(f"[Geocoding Error]: {e}")
        
        return None
    
    def _geocode_google(self, address):
        """Geocode using Google Maps."""
        try:
            import requests
            
            url = (
                f"https://maps.googleapis.com/maps/api/geocode/json?"
                f"address={quote(address)}&key={self.api_key}"
            )
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    location = data['results'][0]['geometry']['location']
                    return {
                        'lat': location['lat'],
                        'lon': location['lng'],
                        'display_name': data['results'][0]['formatted_address']
                    }
                    
        except Exception as e:
            print(f"[Google Geocoding Error]: {e}")
        
        return None
    
    def reverse_geocode(self, lat, lon):
        """Convert coordinates to address."""
        if self.map_provider == 'openstreetmap':
            return self._reverse_geocode_osm(lat, lon)
        elif self.map_provider == 'google' and self.api_key:
            return self._reverse_geocode_google(lat, lon)
        
        return None
    
    def _reverse_geocode_osm(self, lat, lon):
        """Reverse geocode using OpenStreetMap."""
        try:
            import requests
            
            url = (
                f"https://nominatim.openstreetmap.org/reverse?"
                f"lat={lat}&lon={lon}&format=json"
            )
            
            response = requests.get(url, headers={'User-Agent': 'AIris/1.0'})
            
            if response.status_code == 200:
                data = response.json()
                return data.get('display_name', 'Unknown location')
                
        except Exception as e:
            print(f"[Reverse Geocoding Error]: {e}")
        
        return None