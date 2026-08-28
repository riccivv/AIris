import cv2
import time
import math
import numpy as np
from ultralytics import YOLO
from .config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    HAZARD_COVERAGE_THRESHOLD,
    HAZARD_HEIGHT_RATIO,
    AREA_GROWTH_THRESHOLD,
    COOLDOWN_SECONDS,
)


ARROW_COLORS = {
    'navigation': (0, 255, 0),
    'warning': (0, 165, 255),
}


def draw_navigation_arrow(frame, navigation_manager):
    """Draw a navigation arrow pointing in the step's turn direction.

    - Straight steps: straight arrow (continue ahead)
    - Left/right steps: solid arrow pointing left or right
    """
    frame_height, frame_width, _ = frame.shape
    center_x = frame_width // 2
    center_y = frame_height // 2

    nav = navigation_manager
    if not nav or not nav.active_route or nav.current_step_index >= len(nav.active_route):
        return

    step = nav.active_route[nav.current_step_index]
    turn = step.get('turn', '').lower()

    color = ARROW_COLORS['warning'] if nav.is_paused else ARROW_COLORS['navigation']

    if 'left' in turn:
        direction = 'left'
    elif 'right' in turn:
        direction = 'right'
    else:
        # No turn ahead (e.g. final step / arrival) - keep going straight
        _draw_straight_arrow(frame, center_x, center_y, color)
        return

    _draw_directional_arrow(frame, center_x, center_y, direction, color)


def _draw_straight_arrow(frame, center_x, center_y, color):
    """Draw an arrow pointing straight ahead."""
    cv2.arrowedLine(frame, (center_x, center_y + 50), (center_x, center_y - 50),
                   color, 3, tipLength=0.3)


def _draw_directional_arrow(frame, center_x, center_y, direction, color):
    """Draw a solid arrow pointing left or right."""
    if direction == 'left':
        cv2.arrowedLine(frame, (center_x + 50, center_y), (center_x - 50, center_y),
                       color, 3, tipLength=0.3)
    else:
        cv2.arrowedLine(frame, (center_x - 50, center_y), (center_x + 50, center_y),
                       color, 3, tipLength=0.3)


class UIManager:
    def __init__(self, audio_manager, navigation_manager=None, voice_handler=None, model=None):
        self.audio_manager = audio_manager
        self.navigation_manager = navigation_manager
        self.voice_handler = voice_handler
        # Load YOLO eagerly unless a model was provided (e.g. a camera-thread
        # model) to avoid a second blocking load at app startup.
        self.model = model if model is not None else YOLO(YOLO_MODEL_PATH)
        self.notification_banner = ""
        self.banner_expiry = 0
        self.banner_type = "info"  # info, warning, danger, success
        
        # Hazard tracking
        self.last_spoken_time = {}
        self.last_known_coverage = {}
        
        # UI State
        self.show_help = False
        self.help_expiry = 0
        self.animation_start = time.time()
        
        # Colors (BGR format)
        self.colors = {
            'background': (30, 30, 30),
            'header_bg': (20, 20, 20),
            'reading': (255, 165, 0),    # Orange
            'navigation': (0, 255, 0),    # Green
            'warning': (0, 165, 255),     # Yellow
            'danger': (0, 0, 255),        # Red
            'info': (255, 255, 255),      # White
            'success': (0, 255, 0),       # Green
            'card_bg': (40, 40, 40),
            'card_border_pending': (0, 140, 255),  # Orange/Blue
            'card_border_active': (0, 180, 0),      # Green
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
        }
    
    def detect_hazards(self, frame):
        """Run YOLO detection and trigger hazard alerts with distance estimation."""
        frame_height, frame_width, _ = frame.shape
        results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        detected_objects = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                coverage = (box_width * box_height) / (frame_width * frame_height)
                
                # Position detection
                if center_x < frame_width / 3:
                    position = "left"
                elif center_x > (2 * frame_width) / 3:
                    position = "right"
                else:
                    position = "center"
                
                # Rough distance estimation (based on bounding box size)
                if coverage > 0.15:
                    distance = "very close"
                elif coverage > 0.08:
                    distance = "close"
                elif coverage > 0.04:
                    distance = "moderate"
                else:
                    distance = "far"
                
                detected_objects.append({
                    'class': class_name,
                    'bbox': (x1, y1, x2, y2),
                    'position': position,
                    'distance': distance,
                    'coverage': coverage,
                    'center': (center_x, center_y)
                })
                
                # Draw detection with color based on danger level
                color = self._get_danger_color(coverage, box_height / frame_height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw semi-transparent label background
                label = f"{class_name} ({distance})"
                self._draw_label_with_background(frame, label, (x1, y1 - 25), color)
                
                # Trigger alert if hazard is close enough
                if coverage > HAZARD_COVERAGE_THRESHOLD or (box_height / frame_height) > HAZARD_HEIGHT_RATIO:
                    current_time = time.time()
                    time_passed = current_time - self.last_spoken_time.get(class_name, 0)
                    prev_coverage = self.last_known_coverage.get(class_name, 0)
                    is_getting_closer = (coverage - prev_coverage) > AREA_GROWTH_THRESHOLD
                    
                    if time_passed > COOLDOWN_SECONDS or is_getting_closer:
                        message = f"{class_name} {position}."
                        self.audio_manager.speak(message, force=False)
                        
                        self.last_spoken_time[class_name] = current_time
                        self.last_known_coverage[class_name] = coverage
                        self.show_notification(message, "warning", 3.0)
        
        return detected_objects
    
    def _get_danger_color(self, coverage, height_ratio):
        """Return color based on danger level."""
        if coverage > 0.15 or height_ratio > 0.7:
            return self.colors['danger']  # Red for very close
        elif coverage > 0.08 or height_ratio > 0.5:
            return self.colors['warning']  # Yellow for moderate
        else:
            return self.colors['success']  # Green for far
    
    def _draw_label_with_background(self, frame, text, position, color):
        """Draw text with semi-transparent background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        # Draw background
        cv2.rectangle(frame, (x, y - text_height - baseline), 
                     (x + text_width, y + baseline), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    def show_notification(self, message, type="info", duration=3.0):
        """Show notification banner with type-based styling."""
        self.notification_banner = message
        self.banner_type = type
        self.banner_expiry = time.time() + duration
    
    def render(self, frame, location_tracker=None):
        """Render UI elements on frame."""
        frame_height, frame_width, _ = frame.shape
        current_time = time.time()
        current_mode = self.voice_handler.current_mode
        
        # Mode-specific rendering
        if current_mode == "READING":
            self._render_reading_mode(frame)
        elif current_mode == "NAVIGATION":
            self._render_navigation_mode(frame)
        
        # Render header
        self._render_header(frame, current_mode)
        
        # Render location info if provided
        if location_tracker:
            self._render_location_info(frame, location_tracker)
        
        # Render notification banner
        if current_time < self.banner_expiry:
            self._render_notification(frame)
        
        # Render destination card
        self._render_destination_card(frame, current_time)
        
        # Render navigation status if active
        if self.navigation_manager.navigation_started or self.navigation_manager.is_paused:
            self._render_navigation_status(frame, self.navigation_manager)
        
        # Render help overlay if active
        if self.show_help and current_time < self.help_expiry:
            self._render_help_overlay(frame)
    
    def _render_reading_mode(self, frame):
        """Render reading mode specific UI."""
        frame_height, frame_width, _ = frame.shape
        margin_x, margin_y = int(frame_width * 0.15), int(frame_height * 0.15)
        
        # Draw focus rectangle
        cv2.rectangle(frame, (margin_x, margin_y), 
                     (frame_width - margin_x, frame_height - margin_y), 
                     self.colors['reading'], 2)
        
        # Draw corner brackets for modern look
        self._draw_corner_brackets(frame, (margin_x, margin_y), 
                                  (frame_width - margin_x, frame_height - margin_y), 
                                  self.colors['reading'], 20)
        
        # Title
        cv2.putText(frame, "READING MODE", (margin_x + 10, margin_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['reading'], 2)
        
        # Instructions
        instructions = [
            "Say 'READ' to process text",
            "Say 'DESCRIBE' to analyze scene"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (margin_x + 10, margin_y + 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_secondary'], 1)
    
    def _render_navigation_mode(self, frame):
        """Render navigation mode specific UI."""
        frame_height, frame_width, _ = frame.shape
        
        # Draw navigation arrow indicator if route is active
        if self.navigation_manager.active_route:
            self._draw_navigation_arrow(frame)
    
    def _render_header(self, frame, current_mode):
        """Render enhanced header with dynamic sizing."""
        frame_height, frame_width, _ = frame.shape
        header_height = 45
        
        # Create gradient background
        for i in range(header_height):
            alpha = i / header_height
            color = tuple(int(c * alpha) for c in (30, 30, 30))
            cv2.line(frame, (0, i), (frame_width, i), color, 1)
        
        # Draw header line
        header_color = self.colors['navigation'] if current_mode == "NAVIGATION" else self.colors['reading']
        cv2.line(frame, (0, header_height), (frame_width, header_height), header_color, 2)
        
        # Mode indicator with dynamic sizing
        mode_text = f"[{current_mode}]"
        mute_status = " [MUTED]" if self.audio_manager.mute_alerts else ""
        
        # Calculate text size for dynamic badge width
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(mode_text, font, font_scale, thickness)
        
        # Dynamic badge dimensions with padding
        badge_x, badge_y = 10, 10
        badge_padding_x = 15
        badge_padding_y = 5
        badge_w = text_width + (badge_padding_x * 2)
        badge_h = text_height + (badge_padding_y * 2)
        
        # Draw mode badge background
        cv2.rectangle(frame, (badge_x, badge_y), 
                     (badge_x + badge_w, badge_y + badge_h), 
                     header_color, -1)
        
        # Add subtle border to badge
        cv2.rectangle(frame, (badge_x, badge_y), 
                     (badge_x + badge_w, badge_y + badge_h), 
                     (255, 255, 255), 1)
        
        # Center text in badge
        text_x = badge_x + badge_padding_x
        text_y = badge_y + badge_padding_y + text_height
        cv2.putText(frame, mode_text, (text_x, text_y),
                   font, font_scale, (0, 0, 0), thickness)
        
        # Draw mute status if active
        if mute_status:
            mute_x = badge_x + badge_w + 10
            mute_y = badge_y + badge_h - 5
            cv2.putText(frame, mute_status, (mute_x, mute_y),
                       font, 0.5, self.colors['warning'], 1)
        
        # Draw controls hint (right-aligned)
        hint_text = "H:Help | M:Mode | Q:Quit"
        (hint_width, hint_height), _ = cv2.getTextSize(hint_text, font, 0.45, 1)
        hint_x = frame_width - hint_width - 10
        hint_y = badge_y + badge_h - 5
        
        cv2.putText(frame, hint_text, (hint_x, hint_y),
                   font, 0.45, self.colors['text_secondary'], 1)
    
    def _render_location_info(self, frame, location_tracker):
        """Render current location information."""
        if not location_tracker:
            return
        
        frame_height, frame_width, _ = frame.shape
        
        # Get location info
        position = location_tracker.get_position()
        distance = location_tracker.get_distance_from_start()
        
        # Create location text
        location_text = f"Pos: ({position['x']:.1f}, {position['y']:.1f}) | Dist: {distance:.1f}m"
        
        # Draw in top-left corner below header
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, location_text, (10, 70),
                   font, 0.45, self.colors['text_secondary'], 1)
    
    def _render_notification(self, frame):
        """Render enhanced notification banner."""
        frame_height, frame_width, _ = frame.shape
        banner_height = 40
        
        # Get color based on notification type
        color = self.colors.get(self.banner_type, self.colors['info'])
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame_height - banner_height), (frame_width, frame_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw accent line
        cv2.line(frame, (0, frame_height - banner_height), (frame_width, frame_height - banner_height), 
                color, 2)
        
        # Draw notification icon
        icon_map = {
            'info': 'i',
            'warning': '!',
            'danger': '!',
            'success': 'OK'
        }
        icon = icon_map.get(self.banner_type, 'i')
        
        # Draw icon in circle
        icon_center = (30, frame_height - banner_height // 2)
        cv2.circle(frame, icon_center, 12, color, -1)
        cv2.putText(frame, icon, (icon_center[0] - 5, icon_center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw notification text
        cv2.putText(frame, self.notification_banner[:80], (50, frame_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
    
    def draw_destination_card(self, frame):
        """Draw the destination/progress card when a route is active.

        Works without a voice handler (e.g. Streamlit), rendering purely
        from navigation manager state.
        """
        if not self.navigation_manager or not self.navigation_manager.current_destination:
            return
        self._render_destination_card(frame, time.time())

    def _render_destination_card(self, frame, current_time):
        """Render enhanced destination card with progress bar."""
        frame_height, frame_width, _ = frame.shape
        pending = self.voice_handler.pending_destination if self.voice_handler else None
        dest_item = pending or self.navigation_manager.current_destination

        if dest_item:
            dest_name = dest_item["name"]

            if pending:
                dest_text = f"Destination: {dest_name}"
                status_text = "Confirm? Say 'Yes' or 'No'"
                card_color = self.colors['card_border_pending']
                progress = None
            elif self.navigation_manager.active_route and self.navigation_manager.current_step_index < len(self.navigation_manager.active_route):
                step = self.navigation_manager.active_route[self.navigation_manager.current_step_index]
                
                # Use stored remaining distance
                rem_d = int(self.navigation_manager.get_remaining_distance())
                
                # Use stored progress
                progress = self.navigation_manager.get_current_step_progress()
                
                # Check if paused
                if self.navigation_manager.is_paused:
                    dest_text = f"Navigating to: {dest_name}"
                    status_text = f"PAUSED - {step['turn']} in {rem_d}m"
                    card_color = self.colors['warning']
                else:
                    dest_text = f"Navigating to: {dest_name}"
                    status_text = f"{step['turn']} in {rem_d}m"
                    card_color = self.colors['card_border_active']
            else:
                dest_text = f"Destination: {dest_name}"
                status_text = "Route Complete"
                card_color = self.colors['success']
                progress = 1.0
            
            # Draw compact card, bottom center
            card_h = 38
            card_w = max(220, min(int(frame_width * 0.48), 320))
            card_x = (frame_width - card_w) // 2
            card_y = frame_height - card_h - 6

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
                         self.colors['card_bg'], -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Border and accent line
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                         card_color, 1)
            cv2.line(frame, (card_x, card_y), (card_x + card_w, card_y), card_color, 2)

            # Destination name
            cv2.putText(frame, dest_text, (card_x + 10, card_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.colors['text_primary'], 1)

            # Status line
            cv2.putText(frame, status_text, (card_x + 10, card_y + 29),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.36, card_color, 1)

            # Progress bar
            if progress is not None:
                bar_x = card_x + 10
                bar_y = card_y + card_h - 4
                bar_w = card_w - 20
                bar_h = 2
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                            (60, 60, 60), -1)
                filled_w = int(bar_w * progress)
                if filled_w > 0:
                    bar_color = self.colors['warning'] if self.navigation_manager.is_paused else card_color
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                                bar_color, -1)
    
    def _render_navigation_status(self, frame, navigation_manager):
        """Render navigation status indicator."""
        if not navigation_manager:
            return
        
        status = navigation_manager.get_status()
        
        if status['active'] or status['paused']:
            frame_height, frame_width, _ = frame.shape
            
            # Position in top-right corner below header
            status_text = ""
            status_color = self.colors['success']
            
            if status['paused']:
                status_text = "NAVIGATION PAUSED"
                status_color = self.colors['warning']
            elif status['completed']:
                status_text = "ROUTE COMPLETE"
                status_color = self.colors['success']
            else:
                status_text = f"STEP {status['current_step'] + 1}/{status['total_steps']}"
                status_color = self.colors['navigation']
            
            if status_text:
                # Draw status badge
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
                
                badge_x = frame_width - text_width - 30
                badge_y = 50
                badge_w = text_width + 20
                badge_h = text_height + 10
                
                # Background
                cv2.rectangle(frame, (badge_x, badge_y), 
                             (badge_x + badge_w, badge_y + badge_h), 
                             (30, 30, 30), -1)
                
                # Border
                cv2.rectangle(frame, (badge_x, badge_y), 
                             (badge_x + badge_w, badge_y + badge_h), 
                             status_color, 2)
                
                # Text
                cv2.putText(frame, status_text, (badge_x + 10, badge_y + badge_h - 5),
                           font, font_scale, status_color, 1)
    
    def _draw_corner_brackets(self, frame, pt1, pt2, color, length):
        """Draw corner brackets for modern UI look."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)
    
    def _draw_navigation_arrow(self, frame):
        """Draw navigation arrow based on proximity to the next turn."""
        draw_navigation_arrow(frame, self.navigation_manager)
    
    def _render_help_overlay(self, frame):
        """Render help overlay with keyboard shortcuts and voice commands."""
        frame_height, frame_width, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        title = "AIris Help"
        cv2.putText(frame, title, (frame_width // 2 - 50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['info'], 2)
        
        # Commands list
        commands = [
            ("Keyboard:", ""),
            ("  M", "Toggle Mode (Pause/Resume Navigation)"),
            ("  H", "Show/Hide Help"),
            ("  S", "LLM Statistics"),
            ("  Q", "Quit"),
            ("", ""),
            ("Voice Commands:", ""),
            ("  'Go to [place]'", "Navigate to destination"),
            ("  'Cancel Navigation'", "Cancel current route"),
            ("  'Pause Navigation'", "Pause current route"),
            ("  'Resume Navigation'", "Resume paused route"),
            ("  'Read'", "Read text in view"),
            ("  'Describe'", "Describe scene"),
            ("  'Mute/Unmute'", "Toggle alerts"),
            ("  'Switch Mode'", "Toggle between modes"),
        ]
        
        y_offset = 100
        for key, desc in commands:
            if key and desc:
                cv2.putText(frame, f"{key} - {desc}", (frame_width // 4, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 1)
            elif key:
                cv2.putText(frame, key, (frame_width // 4, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['warning'], 1)
            y_offset += 30
        
        # Close hint
        cv2.putText(frame, "Press 'H' to close", (frame_width // 2 - 60, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_secondary'], 1)